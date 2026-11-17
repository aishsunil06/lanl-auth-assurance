"""
Stream LANL auth.txt.gz straight from csr.lanl.gov and write a reduced Parquet
table of lateral-movement candidates.

The raw file is 7.6 GB compressed and roughly 70 GB decompressed. It is never
stored on disk. The HTTP response is decompressed on the fly, filtered line by
line, and only the surviving rows are written out.

Filter (lateral-movement candidates):
  - destination computer differs from source computer
  - the authenticating user is a user account, not a machine account (no "$")
  - the event is a LogOn orientation

Ground truth: an event is labelled red team if (time, src_user, src_computer,
destination_computer) appears in redteam.txt.

Usage:
    python reduce_auth.py <url> <out.parquet> [--max-bytes N] [--max-days N]
"""

import argparse
import gzip
import os
import sys
import time
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

SECONDS_PER_DAY = 86400

SCHEMA = pa.schema(
    [
        ("time", pa.int32()),
        ("day", pa.int16()),
        ("src_user", pa.string()),
        ("dst_user", pa.string()),
        ("src_comp", pa.string()),
        ("dst_comp", pa.string()),
        ("auth_type", pa.string()),
        ("logon_type", pa.string()),
        ("orientation", pa.string()),
        ("success", pa.bool_()),
        ("is_redteam", pa.bool_()),
    ]
)


def load_redteam(path):
    """Return the set of (time, user, src_comp, dst_comp) tuples of known compromise."""
    truth = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t, user, src, dst = line.split(",")
            truth.add((int(t), user, src, dst))
    return truth


def open_stream(url, max_bytes=None):
    """Open the remote gzip as a decompressed text stream."""
    req = urllib.request.Request(url)
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    resp = urllib.request.urlopen(req, timeout=120)
    return gzip.GzipFile(fileobj=resp, mode="rb")


def reduce_auth(url, out_path, redteam_path, max_bytes=None, max_days=None):
    truth = load_redteam(redteam_path)
    print(f"loaded {len(truth)} red team events as ground truth", flush=True)

    cols = {name: [] for name in SCHEMA.names}
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")

    total = kept = matched = malformed = 0
    batch_size = 500_000
    started = time.time()
    stopped_early = False

    stream = open_stream(url, max_bytes)
    try:
        for raw in stream:
            total += 1
            try:
                line = raw.decode("ascii", "replace").rstrip("\n")
                (
                    t_s,
                    src_user,
                    dst_user,
                    src_comp,
                    dst_comp,
                    auth_type,
                    logon_type,
                    orientation,
                    outcome,
                ) = line.split(",")
                t = int(t_s)
            except ValueError:
                malformed += 1
                continue

            day = t // SECONDS_PER_DAY + 1
            if max_days and day > max_days:
                stopped_early = True
                break

            # Lateral movement candidates only.
            if src_comp == dst_comp:
                continue
            if "$@" in src_user or src_user.endswith("$"):
                continue
            if orientation != "LogOn":
                continue

            is_rt = (t, src_user, src_comp, dst_comp) in truth
            matched += is_rt

            cols["time"].append(t)
            cols["day"].append(day)
            cols["src_user"].append(src_user)
            cols["dst_user"].append(dst_user)
            cols["src_comp"].append(src_comp)
            cols["dst_comp"].append(dst_comp)
            cols["auth_type"].append(auth_type)
            cols["logon_type"].append(logon_type)
            cols["orientation"].append(orientation)
            cols["success"].append(outcome == "Success")
            cols["is_redteam"].append(is_rt)
            kept += 1

            if kept % batch_size == 0:
                writer.write_table(pa.table(cols, schema=SCHEMA))
                cols = {name: [] for name in SCHEMA.names}
                rate = total / (time.time() - started)
                print(
                    f"  read {total:,} kept {kept:,} redteam {matched} "
                    f"day {day} ({rate:,.0f} lines/s)",
                    flush=True,
                )
    except EOFError:
        # Expected when a Range request truncates the gzip stream mid-member.
        print("  stream truncated (expected for a ranged sample)", flush=True)
    finally:
        if cols["time"]:
            writer.write_table(pa.table(cols, schema=SCHEMA))
        writer.close()
        stream.close()

    elapsed = time.time() - started
    print(
        f"\ndone in {elapsed/60:.1f} min\n"
        f"  lines read      {total:,}\n"
        f"  rows kept       {kept:,} ({100*kept/max(total,1):.2f}%)\n"
        f"  redteam matched {matched} of {len(truth)}\n"
        f"  malformed       {malformed:,}\n"
        f"  stopped early   {stopped_early}\n"
        f"  output          {out_path} "
        f"({os.path.getsize(out_path)/1e6:.0f} MB)",
        flush=True,
    )
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--redteam", default="data/redteam.txt")
    ap.add_argument("--max-bytes", type=int, default=None)
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    reduce_auth(args.url, args.out, args.redteam, args.max_bytes, args.max_days)


if __name__ == "__main__":
    main()
