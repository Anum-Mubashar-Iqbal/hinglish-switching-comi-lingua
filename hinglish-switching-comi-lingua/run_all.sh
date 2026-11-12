#!/usr/bin/env bash
# Runs the full analysis from the repository root.
# Put LID_train.csv, LID_test.csv, POS_train.csv and POS_test.csv in data/ first.
set -e
for s in load gold align analysis analysis2 analysis3 analysis4 figs; do
  echo "== $s.py"
  python scripts/$s.py > "logs_$s.txt"
done
echo "Done. Results are in the .json/.csv files and the fig/ folder."
