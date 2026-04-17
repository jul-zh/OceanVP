#!/bin/bash

LOG_DIR="/home/yzhidkova/logs"
TMP_FILE=$(mktemp)

patterns=(
  "my_baseline_login_10ep-*.out"
  "my_baseline_kanattn_rbf_login_10ep-*.out"
  "my_baseline_kanattn_rbf_v2_login_10ep-*.out"
  "my_baseline_kanhead_rbf_login_10ep-*.out"
  "my_baseline_kanhead_rbf_login_50ep-*.out"
  "my_baseline_kanhead_rbf_residual_login_10ep-*.out"
  "my_baseline_kanhead_rbf_residual_login_50ep-*.out"
  "my_baseline_kanhead_rbf_residual_dropout0_login_10ep-*.out"
  "my_baseline_kanhead_rbf_residual_centers24_login_10ep-*.out"
  "my_baseline_kanhead_rbf_residual_gamma05_login_10ep-*.out"
  "my_baseline_kanhead_rbf_residual_hidden96_login_10ep-*.out"
  "my_baseline_kanhead_rbf_residual_bestguess_login_10ep-*.out"
  "my_baseline_kanhead_rbf_wide_login_10ep-*.out"
  "my_baseline_kanhead_spline_login_10ep-*.out"
  "my_baseline_kanhead_spline_residual_login_10ep-*.out"
  "my_baseline_kanhead_spline_residual_login_50ep-*.out"
  "my_baseline_kandecoder_gate_rbf_login_10ep-*.out"
  "my_baseline_kanskip_fusion_rbf_login_10ep-*.out"
  "my_baseline_sdefeat_10ep_*.out"
  "my_baseline_sdelight_10ep_*.out"
  "my_baseline_sdelight_50ep_*.out"

  "my_baseline_sdelight_kanhead_rbf_residual_login_10ep-*.out"
  "my_baseline_sdelight_kanhead_rbf_residual_login_50ep-*.out"

  "my_baseline_sdelight_kanhead_rbf_residual_sdeloss_login_10ep-*.out"
  "my_baseline_sdelight_kanhead_rbf_residual_sdeloss_login_50ep-*.out"
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
  my_baseline_login_10ep-*.out)
    short_name="baseline"
    ;;
  my_baseline_kanattn_rbf_login_10ep-*.out)
    short_name="kanattn_rbf"
    ;;
  my_baseline_kanattn_rbf_v2_login_10ep-*.out)
    short_name="kanattn_rbf_v2"
    ;;
  my_baseline_kanhead_rbf_login_10ep-*.out)
    short_name="kanhead_rbf"
    ;;
  my_baseline_kanhead_rbf_login_50ep-*.out)
    short_name="kanhead_rbf_50ep"
    ;;
  my_baseline_kanhead_rbf_residual_login_10ep-*.out)
    short_name="kanhead_rbf_residual"
    ;;
  my_baseline_kanhead_rbf_residual_login_50ep-*.out)
    short_name="kanhead_rbf_residual_50ep"
    ;;
  my_baseline_kanhead_rbf_residual_dropout0_login_10ep-*.out)
    short_name="rbf_resid_dropout0"
    ;;
  my_baseline_kanhead_rbf_residual_centers24_login_10ep-*.out)
    short_name="rbf_resid_centers24"
    ;;
  my_baseline_kanhead_rbf_residual_gamma05_login_10ep-*.out)
    short_name="rbf_resid_gamma05"
    ;;
  my_baseline_kanhead_rbf_residual_hidden96_login_10ep-*.out)
    short_name="rbf_resid_hidden96"
    ;;
  my_baseline_kanhead_rbf_residual_bestguess_login_10ep-*.out)
    short_name="rbf_resid_bestguess"
    ;;
  my_baseline_kanhead_rbf_wide_login_10ep-*.out)
    short_name="kanhead_rbf_wide"
    ;;
  my_baseline_kanhead_spline_login_10ep-*.out)
    short_name="kanhead_spline"
    ;;
  my_baseline_kanhead_spline_residual_login_10ep-*.out)
    short_name="kanhead_spline_residual"
    ;;
  my_baseline_kanhead_spline_residual_login_50ep-*.out)
    short_name="kanhead_spline_residual_50ep"
    ;;
  my_baseline_kandecoder_gate_rbf_login_10ep-*.out)
    short_name="kandecoder_gate_rbf"
    ;;
  my_baseline_kanskip_fusion_rbf_login_10ep-*.out)
    short_name="kanskip_fusion_rbf"
    ;;
  my_baseline_sdefeat_10ep_*.out)
    short_name="sdefeat_10ep"
    ;;
  my_baseline_sdelight_10ep_*.out)
    short_name="sdelight_10ep"
    ;;
  my_baseline_sdelight_50ep_*.out)
    short_name="sdelight_50ep"
    ;;

  my_baseline_sdelight_kanhead_rbf_residual_login_10ep-*.out)
    short_name="sdelight_kanhead_rbf_residual_10ep"
    ;;
  my_baseline_sdelight_kanhead_rbf_residual_login_50ep-*.out)
    short_name="sdelight_kanhead_rbf_residual_50ep"
    ;;

  my_baseline_sdelight_kanhead_rbf_residual_sdeloss_login_10ep-*.out)
    short_name="sdelight_kanhead_rbf_residual_sdeloss_10ep"
    ;;
  my_baseline_sdelight_kanhead_rbf_residual_sdeloss_login_50ep-*.out)
    short_name="sdelight_kanhead_rbf_residual_sdeloss_50ep"
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
  printf "%-24s | %-45s | %-5s | %-12s | %-12s | %-12s | %-12s | %-12s\n" \
    "model" "file" "ep" "train_loss" "vali_loss" "mse" "mae" "rmse"
  printf '%.0s-' {1..160}
  printf "\n"

  sort -t $'\t' -k1,1g "$TMP_FILE" | \
  awk -F '\t' '!seen[$3]++ {
    printf "%-24s | %-45s | %-5s | %-12s | %-12s | %-12s | %-12s | %-12s\n", $2, $3, $4, $5, $6, $1, $7, $8
  }'
}

rm -f "$TMP_FILE"
