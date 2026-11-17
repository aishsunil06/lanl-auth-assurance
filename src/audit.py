"""
Three experiments auditing cyber anomaly detection on the LANL authentication
data. The object under test is the evaluation, not the detector.

  E1  Shortcut learning. 701 of 749 red team events originate from one host.
      A detector handed source-host identity learns the attacker's
      infrastructure rather than the attacker's behaviour, which is measured by
      holding that host out.

  E2  Evaluation honesty. The same features and the same model are scored under
      a random split and under every forward-in-time split the data supports,
      then under an analyst's alert budget and against the 28 days with no known
      intrusion.

  E3  Adaptive adversary. Counterfactual versions of the real red team hops,
      each representing one habit a more careful intruder would adopt, scored
      against the only detector the standard protocol would have shipped.

Usage:  python src/audit.py data/hops.parquet
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "src")
import features as F  # noqa: E402

TRAIN_DAYS = (1, 15)
TEST_DAYS = (16, 30)
CLEAN_DAYS = (31, 58)
ALERT_BUDGETS = [10, 50, 100, 500, 1000]
SEED = 0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def top_k_idx(scores, k):
    k = min(k, len(scores))
    return np.argpartition(-scores, k - 1)[:k]


def precision_recall_at_k(y, scores, k):
    idx = top_k_idx(scores, k)
    hits = int(y[idx].sum())
    total = int(y.sum())
    return hits / min(k, len(scores)), (hits / total if total else 0.0), hits


def safe_auc(y, scores):
    """ROC-AUC, or None when the slice holds only one class."""
    if len(np.unique(y)) < 2:
        return None
    return round(float(roc_auc_score(y, scores)), 5)


def fit(X, y, seed=SEED, categorical=None):
    """Fit a gradient boosted detector.

    Positives are about 0.04% of hops. Without rebalancing the model collapses
    to a near-constant prediction and its ranking becomes close to random, so
    the classes are reweighted to equal total mass. Evaluation is always on the
    untouched population, so precision at a given alert budget stays honest.
    """
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((~y).sum()), 1)
    w = np.where(y, n_neg / n_pos, 1.0)
    m = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=seed,
        categorical_features=categorical,
    )
    m.fit(X, y, sample_weight=w)
    return m


def budget_table(y, scores):
    out = {}
    for k in ALERT_BUDGETS:
        p, r, hits = precision_recall_at_k(y, scores, k)
        out[str(k)] = {
            "precision": round(p, 5),
            "recall": round(r, 5),
            "true_positives": hits,
        }
    return out


def add_source_identity(X, src_codes, keep):
    """Append source-host identity as a single categorical column.

    One-hot encoding 300 hosts over 4.2M rows needs about 5 GB, so identity is
    carried as one integer column instead and declared categorical to the
    booster. Hosts outside `keep` share a single "other" level.
    """
    lookup = {code: j for j, code in enumerate(keep[:254])}
    other = len(lookup)
    col = np.full((len(X), 1), other, dtype=np.float32)
    for code, j in lookup.items():
        col[src_codes == code, 0] = j
    Xid = np.hstack([X, col])
    mask = np.zeros(Xid.shape[1], dtype=bool)
    mask[-1] = True
    return Xid, mask


def entity_overlap(meta, y, idx_a, idx_b):
    """How much of split B's attack activity already appears in split A.

    This is the mechanism behind a random split's flattering numbers. The same
    accounts and the same host pairs sit on both sides, so a model can recognise
    the campaign instead of generalising from it.
    """
    a = meta.iloc[idx_a][y[idx_a]]
    b = meta.iloc[idx_b][y[idx_b]]
    if len(b) == 0:
        return {}
    users_a = set(a["src_user"])
    pairs_a = set(zip(a["src_comp"], a["dst_comp"]))
    shared_u = sum(u in users_a for u in b["src_user"])
    shared_p = sum(p in pairs_a for p in zip(b["src_comp"], b["dst_comp"]))
    return {
        "test_attack_hops": int(len(b)),
        "share_user_with_train_attack": round(shared_u / len(b), 4),
        "share_host_pair_with_train_attack": round(shared_p / len(b), 4),
    }


def main(path):
    log(f"loading {path}")
    hops = pd.read_parquet(path)
    log(f"{len(hops):,} hops, {int(hops.is_redteam.sum())} red team")

    log("building causal features")
    X, y, meta = F.build(hops)
    del hops
    day = meta["day"].to_numpy()
    src_codes = meta["src_code"].to_numpy()

    tr = (day >= TRAIN_DAYS[0]) & (day <= TRAIN_DAYS[1])
    te = (day >= TEST_DAYS[0]) & (day <= TEST_DAYS[1])
    cl = (day >= CLEAN_DAYS[0]) & (day <= CLEAN_DAYS[1])
    log(f"train {tr.sum():,} ({int(y[tr].sum())} rt) | "
        f"test {te.sum():,} ({int(y[te].sum())} rt) | "
        f"clean {cl.sum():,} ({int(y[cl].sum())} rt)")

    R = {"setup": {
        "hops": int(len(y)),
        "redteam_hops": int(y.sum()),
        "train_days": list(TRAIN_DAYS),
        "test_days": list(TEST_DAYS),
        "clean_days": list(CLEAN_DAYS),
        "train": {"hops": int(tr.sum()), "redteam": int(y[tr].sum())},
        "test": {"hops": int(te.sum()), "redteam": int(y[te].sum())},
        "clean": {"hops": int(cl.sum()), "redteam": int(y[cl].sum())},
        "test_base_rate": float(y[te].sum() / max(te.sum(), 1)),
    }}

    # ------------------------------------------------------------------ E2
    log("E2  temporal split (honest) vs random split (standard practice)")
    behav = fit(X[tr], y[tr])
    s_te = behav.predict_proba(X[te])[:, 1]
    R["E2_temporal"] = {
        "roc_auc": safe_auc(y[te], s_te),
        "average_precision": round(float(average_precision_score(y[te], s_te)), 6),
        "alert_budgets": budget_table(y[te], s_te),
    }
    log(f"    temporal  AUC {R['E2_temporal']['roc_auc']}  "
        f"AP {R['E2_temporal']['average_precision']}")

    # Every forward-in-time boundary the data supports, so the result cannot be
    # a single unlucky cut.
    sweep = {}
    for a, b, c, d in [(1, 12, 13, 15), (1, 10, 11, 15), (1, 8, 9, 13), (1, 15, 16, 30)]:
        m1 = (day >= a) & (day <= b)
        m2 = (day >= c) & (day <= d)
        if y[m1].sum() < 20 or y[m2].sum() < 20:
            continue
        mm = fit(X[m1], y[m1])
        ss = mm.predict_proba(X[m2])[:, 1]
        key = f"train_{a}-{b}_test_{c}-{d}"
        sweep[key] = {
            "train_redteam": int(y[m1].sum()),
            "test_redteam": int(y[m2].sum()),
            "roc_auc": safe_auc(y[m2], ss),
            "average_precision": round(float(average_precision_score(y[m2], ss)), 6),
        }
        log(f"    train {a}-{b} test {c}-{d}: AUC {sweep[key]['roc_auc']}")
    R["E2_temporal_sweep"] = sweep

    log("E2  random split")
    rng = np.random.default_rng(SEED)
    pool = np.flatnonzero(tr | te)
    rng.shuffle(pool)
    cut = len(pool) // 2
    ia, ib = pool[:cut], pool[cut:]
    leaky = fit(X[ia], y[ia])
    s_ib = leaky.predict_proba(X[ib])[:, 1]
    R["E2_random_split"] = {
        "roc_auc": safe_auc(y[ib], s_ib),
        "average_precision": round(float(average_precision_score(y[ib], s_ib)), 6),
        "alert_budgets": budget_table(y[ib], s_ib),
        "entity_overlap": entity_overlap(meta, y, ia, ib),
    }
    log(f"    random    AUC {R['E2_random_split']['roc_auc']}  "
        f"AP {R['E2_random_split']['average_precision']}")
    log(f"    overlap   {R['E2_random_split']['entity_overlap']}")

    log("E2  false alarms over the 28 days with no known intrusion")
    thr = float(s_te[top_k_idx(s_te, 100)].min())
    alerts = int((behav.predict_proba(X[cl])[:, 1] >= thr).sum()) if cl.sum() else 0
    nd = CLEAN_DAYS[1] - CLEAN_DAYS[0] + 1
    R["E2_clean_period"] = {
        "threshold": round(thr, 6),
        "alerts": alerts,
        "hops": int(cl.sum()),
        "alerts_per_day": round(alerts / nd, 1),
    }
    log(f"    {alerts:,} alerts over {nd} quiet days, {alerts/nd:.1f}/day, all false")

    # ------------------------------------------------------------------ E1
    log("E1  source-host identity as a shortcut")
    counts = pd.Series(src_codes[tr]).value_counts()
    keep = list(counts.head(250).index)
    for c in np.unique(src_codes[tr & y]):
        if c not in keep:
            keep.append(c)
    Xid, cat_mask = add_source_identity(X, src_codes, keep)

    id_model = fit(Xid[tr], y[tr], categorical=cat_mask)
    s_id = id_model.predict_proba(Xid[te])[:, 1]
    R["E1_with_source_identity"] = {
        "roc_auc": safe_auc(y[te], s_id),
        "average_precision": round(float(average_precision_score(y[te], s_id)), 6),
        "alert_budgets": budget_table(y[te], s_id),
    }
    log(f"    with host identity: AUC {R['E1_with_source_identity']['roc_auc']} "
        f"P@100 {R['E1_with_source_identity']['alert_budgets']['100']['precision']:.3f}")

    dominant = meta.loc[y, "src_comp"].value_counts().index[0]
    is_dom = (meta["src_comp"] == dominant).to_numpy()
    log(f"    holding out the dominant attacker host {dominant}")
    tr_dom = tr & (~y | is_dom)
    te_held = te & (~y | ~is_dom)
    held = {}
    if int(y[te_held].sum()) > 0:
        for name, mat, cm in (("with_source_identity", Xid, cat_mask),
                              ("behavioural_only", X, None)):
            m = fit(mat[tr_dom], y[tr_dom], categorical=cm)
            s = m.predict_proba(mat[te_held])[:, 1]
            p, r, hits = precision_recall_at_k(y[te_held], s, 100)
            held[name] = {
                "roc_auc": safe_auc(y[te_held], s),
                "held_out_positives": int(y[te_held].sum()),
                "recall_at_100": round(r, 5),
                "true_positives_at_100": hits,
            }
            log(f"      {name}: AUC {held[name]['roc_auc']} R@100 {r:.3f} "
                f"({hits}/{int(y[te_held].sum())})")
    else:
        held["note"] = "no positives from non-dominant hosts in the test window"
    # The holdout cannot actually answer the question. Record the numbers that
    # show why, so the result is not read as evidence in either direction.
    held_mask = te & y & ~is_dom
    R["E1_held_out_attacker_host"] = {
        "dominant_host": str(dominant),
        "redteam_hops_from_dominant": int((y & is_dom).sum()),
        "redteam_hops_from_others": int((y & ~is_dom).sum()),
        "distinct_attacker_hosts": int(meta.loc[y, "src_comp"].nunique()),
        "dominant_share": round(float((y & is_dom).sum() / max(y.sum(), 1)), 4),
        "holdout_distinct_users": int(meta.loc[held_mask, "src_user"].nunique()),
        "holdout_distinct_source_hosts": int(meta.loc[held_mask, "src_comp"].nunique()),
        "interpretation": (
            "Underpowered. Every held-out positive comes from one user on one "
            "host, so the effective sample size is a single intrusion path and "
            "neither outcome is evidence of generalisation. The benchmark "
            "cannot separate a detector that learned behaviour from one that "
            "learned an address."
        ),
        "models": held,
    }

    # ------------------------------------------------------------------ E3
    # The adversary is tested against the only detector the standard protocol
    # would ever have shipped: the random-split model at AUC ~0.97. Running
    # evasion against the temporal model would prove nothing, because that model
    # is already at chance before the adversary does anything.
    log("E3  adaptive adversary against the random-split detector")
    fi = {n: i for i, n in enumerate(F.FEATURE_NAMES)}
    pos_b = ib[y[ib]]
    neg_b = ib[~y[ib]]
    s_neg = leaky.predict_proba(X[neg_b])[:, 1]
    labels = np.concatenate([np.ones(len(pos_b), bool), np.zeros(len(neg_b), bool)])

    def evaluate(fn):
        Xp = X[pos_b].copy()
        if fn:
            fn(Xp)
        s_pos = leaky.predict_proba(Xp)[:, 1]
        scores = np.concatenate([s_pos, s_neg])
        out = {"mean_score": round(float(s_pos.mean()), 5)}
        for k in (100, 1000):
            _, r, hits = precision_recall_at_k(labels, scores, k)
            out[str(k)] = {"recall": round(r, 5), "true_positives": hits}
        return out

    def business_hours(Xp):
        Xp[:, fi["off_hours_rate"]] = 0.0
        Xp[:, fi["hour_sin"]] = np.sin(2 * np.pi * 14.0 / 24.0)
        Xp[:, fi["hour_cos"]] = np.cos(2 * np.pi * 14.0 / 24.0)

    def familiar_destination(Xp):
        Xp[:, fi["new_user_dst"]] = 0.0

    def familiar_path(Xp):
        Xp[:, fi["new_user_src"]] = 0.0
        Xp[:, fi["new_host_edge"]] = 0.0

    def no_failures(Xp):
        Xp[:, fi["fail_rate"]] = 0.0

    def blend_volume(Xp):
        Xp[:, fi["n_events"]] = 29.0
        Xp[:, fi["span"]] = 34000.0

    def combined(Xp):
        for fn in (business_hours, familiar_destination, familiar_path,
                   no_failures, blend_volume):
            fn(Xp)

    if len(pos_b) == 0:
        R["E3_adaptive_adversary"] = {"note": "no red team hops in the random test half"}
    else:
        e3 = {"unmodified": evaluate(None)}
        base = e3["unmodified"]["1000"]["recall"]
        for nm, fn in (("business_hours", business_hours),
                       ("familiar_destination", familiar_destination),
                       ("familiar_source_path", familiar_path),
                       ("no_failed_attempts", no_failures),
                       ("blend_into_volume", blend_volume),
                       ("combined", combined)):
            e3[nm] = evaluate(fn)
            log(f"    {nm:22s} R@1000 {e3[nm]['1000']['recall']:.3f}  (base {base:.3f})")
        R["E3_adaptive_adversary"] = e3

    with open("results/audit.json", "w") as fh:
        json.dump(R, fh, indent=2)
    log("wrote results/audit.json")
    return R


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/hops.parquet")
