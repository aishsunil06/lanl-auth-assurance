"""
Causal features for hop-level LANL authentication records.

A "hop" is one (user, source host, destination host, day) row produced by
aggregate.py.

History features are built from *strictly earlier days only*. Same-day activity
never contributes to a hop's own history features, so nothing leaks backwards
from the moment under test. This is deliberate: the obvious framing of this
dataset, "has this user ever reached this host", is trivially leaky if computed
across all 58 days at once, and that leak is one of the things the audit is
about.
"""

import numpy as np
import pandas as pd

HISTORY_FEATURES = [
    "new_user_dst",      # user has not reached this destination on any prior day
    "new_host_edge",     # this source->destination edge is unseen
    "new_user_src",      # user has not authenticated from this source before
    "user_dst_degree",   # distinct destinations this user has reached so far
    "src_out_degree",    # distinct destinations reached from this source so far
    "dst_in_degree",     # distinct sources that have reached this destination
    "user_active_days",  # days on which this user has appeared so far
]

SAMEDAY_FEATURES = [
    "n_events",
    "fail_rate",
    "off_hours_rate",
    "span",
    "n_auth_types",
    "n_logon_types",
    "any_success",
    "hour_sin",
    "hour_cos",
]

FEATURE_NAMES = HISTORY_FEATURES + SAMEDAY_FEATURES


def build(hops):
    """Return (X, y, meta) for a hop table sorted by day.

    hops must contain the aggregate.py output columns.
    """
    hops = hops.sort_values(["day", "first_time"], kind="stable").reset_index(drop=True)

    user_c, _ = hops["src_user"].factorize()
    src_c, _ = hops["src_comp"].factorize()
    dst_c, _ = hops["dst_comp"].factorize()
    user_c = user_c.astype(np.int64)
    src_c = src_c.astype(np.int64)
    dst_c = dst_c.astype(np.int64)

    n = len(hops)
    day_a = hops["day"].to_numpy(np.int32)

    M = 1 << 20
    seen_user_dst, seen_edge, seen_user_src = set(), set(), set()
    user_dsts, src_dsts, dst_srcs, user_days = {}, {}, {}, {}

    H = np.zeros((n, len(HISTORY_FEATURES)), dtype=np.float32)
    pending = []           # rows of the day currently being read
    current_day = day_a[0] if n else None

    def commit(rows):
        for i in rows:
            u, s, d = user_c[i], src_c[i], dst_c[i]
            seen_user_dst.add(u * M + d)
            seen_edge.add(s * M + d)
            seen_user_src.add(u * M + s)
            user_dsts.setdefault(u, set()).add(d)
            src_dsts.setdefault(s, set()).add(d)
            dst_srcs.setdefault(d, set()).add(s)
            user_days.setdefault(u, set()).add(day_a[i])

    for i in range(n):
        if day_a[i] != current_day:
            commit(pending)
            pending = []
            current_day = day_a[i]

        u, s, d = user_c[i], src_c[i], dst_c[i]
        H[i, 0] = (u * M + d) not in seen_user_dst
        H[i, 1] = (s * M + d) not in seen_edge
        H[i, 2] = (u * M + s) not in seen_user_src
        H[i, 3] = len(user_dsts.get(u, ()))
        H[i, 4] = len(src_dsts.get(s, ()))
        H[i, 5] = len(dst_srcs.get(d, ()))
        H[i, 6] = len(user_days.get(u, ()))
        pending.append(i)

    commit(pending)

    hour = hops["first_hour"].to_numpy(np.float32)
    same = np.column_stack(
        [
            hops["n_events"].to_numpy(np.float32),
            hops["fail_rate"].to_numpy(np.float32),
            hops["off_hours_rate"].to_numpy(np.float32),
            hops["span"].to_numpy(np.float32),
            hops["n_auth_types"].to_numpy(np.float32),
            hops["n_logon_types"].to_numpy(np.float32),
            hops["any_success"].to_numpy(np.float32),
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
        ]
    ).astype(np.float32)

    X = np.hstack([H, same])
    y = hops["is_redteam"].to_numpy(bool)
    meta = hops[["day", "src_user", "src_comp", "dst_comp", "first_time"]].copy()
    meta["src_code"] = src_c
    return X, y, meta
