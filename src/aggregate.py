"""
Collapse raw lateral authentication events into one row per
(user, source host, destination host, day).

Two reasons for this unit. It is what an analyst actually triages: the question
is "did this account make this hop today", not "what happened in packet
1,204,118". And it keeps the whole study in memory on an ordinary laptop, which
the ~108M raw lateral events would not.

The file is processed one day at a time, so peak memory is a single day of
events rather than the whole dataset.

Usage:  python src/aggregate.py data/auth_lateral.parquet data/hops.parquet
"""

import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BUSINESS_START, BUSINESS_END = 8, 18
SECONDS_PER_DAY = 86400

KEYS = ["src_user", "src_comp", "dst_comp", "day"]


def aggregate_day(frame):
    hour = (frame["time"] % SECONDS_PER_DAY) / 3600.0
    frame = frame.assign(
        _off=((hour < BUSINESS_START) | (hour >= BUSINESS_END)),
        _fail=~frame["success"],
        _hour=hour,
    )
    g = frame.groupby(KEYS, observed=True, sort=False)
    out = g.agg(
        n_events=("time", "size"),
        n_failures=("_fail", "sum"),
        any_success=("success", "max"),
        first_time=("time", "min"),
        last_time=("time", "max"),
        first_hour=("_hour", "min"),
        off_hours_events=("_off", "sum"),
        n_auth_types=("auth_type", "nunique"),
        n_logon_types=("logon_type", "nunique"),
        is_redteam=("is_redteam", "max"),
    ).reset_index()
    return out


def main(src, dst):
    pf = pq.ParquetFile(src)
    print(f"{pf.metadata.num_rows:,} rows in {pf.metadata.num_row_groups} row groups",
          flush=True)

    parts = []
    buffer = []
    current_day = None
    started = time.time()
    rows_in = 0

    def flush():
        if not buffer:
            return
        frame = pd.concat(buffer, ignore_index=True)
        parts.append(aggregate_day(frame))
        buffer.clear()

    for batch in pf.iter_batches(batch_size=1_000_000):
        frame = batch.to_pandas()
        rows_in += len(frame)
        for day, chunk in frame.groupby("day", sort=True, observed=True):
            if current_day is None:
                current_day = day
            if day != current_day:
                flush()
                current_day = day
                print(
                    f"  day {day}  in {rows_in:,}  out "
                    f"{sum(len(p) for p in parts):,}  "
                    f"{time.time()-started:.0f}s",
                    flush=True,
                )
            buffer.append(chunk)
    flush()

    hops = pd.concat(parts, ignore_index=True)
    del parts
    hops = hops.sort_values(["day", "first_time"], kind="stable").reset_index(drop=True)
    hops["fail_rate"] = hops["n_failures"] / hops["n_events"]
    hops["off_hours_rate"] = hops["off_hours_events"] / hops["n_events"]
    hops["span"] = hops["last_time"] - hops["first_time"]

    hops.to_parquet(dst, compression="zstd", index=False)
    print(
        f"\nwrote {dst}\n"
        f"  events in   {rows_in:,}\n"
        f"  hops out    {len(hops):,} ({rows_in/max(len(hops),1):.1f}x reduction)\n"
        f"  redteam hops {int(hops.is_redteam.sum())}\n"
        f"  days        {hops.day.min()} - {hops.day.max()}\n"
        f"  elapsed     {(time.time()-started)/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
