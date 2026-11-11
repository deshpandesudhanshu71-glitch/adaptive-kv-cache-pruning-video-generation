#!/bin/bash
# Every cache method at 69 frames (18 latent frames), dog prompt, seed 0.
# Run from the repo folder with the NOVA environment active.
set -e

while read -r args; do
  echo "=== $args  $(date +%H:%M)"
  python -W ignore run_method.py $args
done <<'LIST'
--method full --analyse
--method full --tag full_repeat
--method full --analyse --tag analysis
--method window --n 1
--method window --n 2
--method window --n 4
--method window --n 8
--method sinkwindow --n 1
--method sinkwindow --n 2
--method sinkwindow --n 4
--method h2o --n 2
--method h2o --n 4
--method h2o --n 8
--method full --bits 8
--method full --bits 4
--method window --n 4 --bits 8
--method adaptive --tau 0.8
--method adaptive --tau 0.9
--method adaptive --tau 0.9 --bits 8
--method adaptive --tau 0.9 --recheck 2
--method adaptive --tau 0.9 --recheck 4
LIST

python -W ignore decode_eval.py --prompt dog --length 18
python quality.py --prompt dog --length 18
python analyse.py --prompt dog --length 18
