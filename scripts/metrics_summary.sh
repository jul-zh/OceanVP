#!/bin/bash

LOG_DIR="/home/yzhidkova/logs"

printf "%-30s | %-3s | %-7s | %-7s | %-7s | %-7s | %-7s | %-7s\n" \
  "file" "ep" "train_loss" "vali_loss" "mse" "mae" "rmse" "final_result"
printf '%.0s-' {1..145}
printf "\n"

for f in \
  "$LOG_DIR"/my_baseline*.out \
  "$LOG_DIR"/my_baseline_kanattn*.out \
  "$LOG_DIR"/my_baseline_kanhead*.out; do
  [ -f "$f" ] || continue

  epoch_line=$(grep "Epoch:" "$f" | tail -n 1)
  test_line=$(grep "mse:" "$f" | tail -n 1)
  final_line=$(grep "Final result:" "$f" | tail -n 1)

  ep=$(echo "$epoch_line" | sed -n 's/.*Epoch: \([0-9]\+\),.*/\1/p')
  train_loss=$(echo "$epoch_line" | sed -n 's/.*Train Loss: \([0-9.eE+-]\+\).*/\1/p')
  vali_loss=$(echo "$epoch_line" | sed -n 's/.*Vali Loss: \([0-9.eE+-]\+\).*/\1/p')

  mse=$(echo "$test_line" | sed -n 's/.*mse:\([0-9.eE+-]\+\),.*/\1/p')
  mae=$(echo "$test_line" | sed -n 's/.*mae:\([0-9.eE+-]\+\),.*/\1/p')
  rmse=$(echo "$test_line" | sed -n 's/.*rmse:\([0-9.eE+-]\+\).*/\1/p')

  final_result=$(echo "$final_line" | sed -n 's/.*Final result: \([0-9.eE+-]\+\).*/\1/p')

  printf "%-45s | %-5s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s\n" \
    "$(basename "$f")" \
    "${ep:--}" \
    "${train_loss:--}" \
    "${vali_loss:--}" \
    "${mse:--}" \
    "${mae:--}" \
    "${rmse:--}" \
    "${final_result:--}"
done | sort
