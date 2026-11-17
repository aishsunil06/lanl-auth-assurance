#!/usr/bin/env bash
# End-to-end reproduction. AUTH_URL comes from the download form at
# https://csr.lanl.gov/data/cyber1/ and is specific to your registration.
set -euo pipefail
: "${AUTH_URL:?set AUTH_URL to your csr.lanl.gov data-fence link for auth.txt.gz}"
mkdir -p data results
curl -s -L "${AUTH_URL%auth.txt.gz}redteam.txt.gz" -o data/redteam.txt.gz
gunzip -kf data/redteam.txt.gz
python src/reduce_auth.py "$AUTH_URL" data/auth_lateral.parquet
python src/aggregate.py data/auth_lateral.parquet data/hops.parquet
python src/audit.py data/hops.parquet
python src/plots.py
