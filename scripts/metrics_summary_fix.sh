#!/bin/bash

LOG_DIR="/home/yzhidkova/logs"
TMP_FILE=$(mktemp)

patterns=(
  "mb_fix_50ep-*.out"
  "mb_sdelight_fix_50ep-*.out"
  "mb_kanhead_rbf_residual_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_fix_50ep-*.out"
  "mb_cotere_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_fix_10ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_fix_10ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_v2_fix_10ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_v2_fix_50ep-*.out"
  "mb_strongenc_fix_10ep-*.out"
  "mb_strongenc_fix_50ep-*.out"
  "mb_strongenc_kanhead_fix_10ep-*.out"
  "mb_strongenc_kanhead_fix_50ep-*.out"
  "mb_strongenc_sdelight_kanhead_sdeloss_fix_10ep-*.out"
  "mb_strongenc_sdelight_kanhead_sdeloss_fix_50ep-*.out"
)

for pattern in "${patterns[@]}"; do
  for f in "$LOG_DIR"/$pattern; do
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

    [ -n "$mse" ] || continue

    name=$(basename "$f")
    short_name="$name"

    case "$name" in
      mb_fix_50ep-*.out)
        short_name="baseline_fix_50ep"
        ;;
      mb_sdelight_fix_50ep-*.out)
        short_name="sdelight_fix_50ep"
        ;;
      mb_kanhead_rbf_residual_fix_50ep-*.out)
        short_name="kanhead_rbf_residual_fix_50ep"
        ;;
      mb_sdelight_kanhead_rbf_residual_fix_50ep-*.out)
        short_name="sdelight_kanhead_rbf_residual_fix_50ep"
        ;;
      mb_cotere_fix_50ep-*.out)
        short_name="cotere_fix_50ep"
        ;; 
      mb_sdelight_kanhead_rbf_residual_sdeloss_fix_10ep-*.out)
        short_name="sdeloss_fix_10ep"
        ;;
      mb_sdelight_kanhead_rbf_residual_sdeloss_fix_50ep-*.out)
        short_name="sdeloss_fix_50ep"
        ;;
      mb_sdelight_kanhead_rbf_residual_sdeloss_bins_v2_fix_10ep-*.out)
        short_name="sdeloss_bins_v2_fix_10ep"
        ;;
      mb_sdelight_kanhead_rbf_residual_sdeloss_bins_v2_fix_50ep-*.out)
        short_name-"sdeloss_bins_v2_fix_50ep"
        ;;
      mb_strongenc_fix_10ep-*.out)
        short_name="strongenc_fix_10ep"
        ;;
      mb_strongenc_fix_50ep-*.out)
        short_name="strongenc_fix_50ep"
        ;;
      mb_strongenc_kanhead_fix_10ep-*.out)
        short_name="strongenc_kanhead_fix_10ep"
        ;;
      mb_strongenc_kanhead_fix_50ep-*.out)
        short_name="strongenc_kanhead_fix_50ep"
        ;;
      mb_strongenc_sdelight_kanhead_sdeloss_fix_10ep-*.out)
        short_name="strongenc_sdelight_kanhead_sdeloss_fix_10ep"
        ;;
      mb_strongenc_sdelight_kanhead_sdeloss_fix_50ep-*.out)
        short_name="strongenc_sdelight_kanhead_sdeloss_fix_50ep"
        ;;
   esac

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$mse" \
      "$short_name" \
      "$name" \
      "${ep:--}" \
      "${train_loss:--}" \
      "${vali_loss:--}" \
      "${mae:--}" \
      "${rmse:--}" \
      "${final_result:--}" >> "$TMP_FILE"
  done
done

{
  printf "%-36s | %-5s | %-12s | %-12s | %-12s | %-12s | %-12s\n" \
    "model" "ep" "train_loss" "vali_loss" "mse" "mae" "rmse"
  printf '%.0s-' {1..122}
  printf "\n"

  sort -t $'\t' -k1,1g "$TMP_FILE" | \
  awk -F '\t' '!seen[$3]++ {
    printf "%-36s | %-5s | %-12s | %-12s | %-12s | %-12s | %-12s\n", $2, $4, $5, $6, $1, $7, $8
  }'
}

rm -f "$TMP_FILE"
