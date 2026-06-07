#!/bin/bash
# 8 cells x 3 seeds staged-census sweep -> staged_tv.csv
echo "file,seed,tv" > staged_tv.csv
for f in data_kl_o0.01.pth data_balance_o0.01.pth data_kl_o0.015.pth data_balance_o0.015.pth data_kl_o0.02.pth data_balance_o0.02.pth data_kl_o0.025.pth data_balance_o0.025.pth; do
  for s in 1 2 3; do
    tv=$(/home/xuda/.envs/torch/bin/python staged_census.py $f $s 2>/dev/null | grep -oP 'TV\(staged census, referee\) = \K[0-9.]+')
    echo "$f,$s,$tv" >> staged_tv.csv
    echo "$f seed $s: $tv"
  done
done
echo SWEEP-DONE
