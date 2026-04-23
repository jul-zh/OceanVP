#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/home/yzhidkova/logs"

patterns=(
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma15_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma2_horizon_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma2_c32_fix_50ep-*.out"

  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_sharp_lightreg_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_sharp_capacity_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_sharp_nodrop_fix_50ep-*.out"

  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_wide_nodrop_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_big_capacity_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_centers_fix_50ep-*.out"
)

printf "%-42s | %-6s | %-12s | %-12s | %-12s | %-12s | %-12s | %s\n" \
  "model" "ep" "train_loss" "vali_loss" "mse" "mae" "rmse" "file"
printf '%s\n' "-----------------------------------------------------------------------------------------------------------------------------------------------"

extract_last_match() {
  local regex="$1"
  local file="$2"
  grep -E "$regex" "$file" | tail -n 1 || true
}

for pattern in "${patterns[@]}"; do
  shopt -s nullglob
  files=( ${LOG_DIR}/${pattern} )
  shopt -u nullglob

  [[ ${#files[@]} -eq 0 ]] && continue

  # Берем самый свежий лог этого типа
  latest_file="$(ls -1t "${files[@]}" | head -n 1)"
  base_name="$(basename "$latest_file")"

  case "$base_name" in
    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma15_fix_50ep-*.out)
      short_name="sdeloss_bins_gamma15_fix_50ep"
      ;;
    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma2_horizon_fix_50ep-*.out)
      short_name="sdeloss_bins_gamma2_horizon_fix_50ep"
      ;;
    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_gamma2_c32_fix_50ep-*.out)
      short_name="sdeloss_bins_gamma2_c32_fix_50ep"
      ;;

    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_sharp_lightreg_fix_50ep-*.out)
      short_name="sdeloss_bins_sharp_lightreg_fix_50ep"
      ;;
    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_sharp_capacity_fix_50ep-*.out)
      short_name="sdeloss_bins_sharp_capacity_fix_50ep"
      ;;
    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_sharp_nodrop_fix_50ep-*.out)
      short_name="sdeloss_bins_sharp_nodrop_fix_50ep"
      ;;

    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_wide_nodrop_fix_50ep-*.out)
      short_name="sdeloss_bins_wide_nodrop_fix_50ep"
      ;;
    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_big_capacity_fix_50ep-*.out)
      short_name="sdeloss_bins_big_capacity_fix_50ep"
      ;;
    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_centers_fix_50ep-*.out)
      short_name="sdeloss_bins_many_centers_fix_50ep"
      ;;
    *)
      short_name="$base_name"
      ;;
  esac

  epoch_line="$(extract_last_match 'Epoch: [0-9]+' "$latest_file")"
  metric_line="$(extract_last_match 'mse:[0-9eE+.-]+, mae:[0-9eE+.-]+, rmse:[0-9eE+.-]+' "$latest_file")"

  ep="$(echo "$epoch_line" | sed -n 's/.*Epoch: \([0-9]\+\).*/\1/p')"
  train_loss="$(echo "$epoch_line" | sed -n 's/.*Train Loss: \([0-9.eE+-]\+\).*/\1/p')"
  vali_loss="$(echo "$epoch_line" | sed -n 's/.*Vali Loss: \([0-9.eE+-]\+\).*/\1/p')"

  mse="$(echo "$metric_line" | sed -n 's/.*mse:\([0-9.eE+-]\+\), mae:.*/\1/p')"
  mae="$(echo "$metric_line" | sed -n 's/.*mae:\([0-9.eE+-]\+\), rmse:.*/\1/p')"
  rmse="$(echo "$metric_line" | sed -n 's/.*rmse:\([0-9.eE+-]\+\).*/\1/p')"

  printf "%-42s | %-6s | %-12s | %-12s | %-12s | %-12s | %-12s | %s\n" \
    "${short_name}" "${ep:-?}" "${train_loss:-?}" "${vali_loss:-?}" "${mse:-?}" "${mae:-?}" "${rmse:-?}" "${base_name}"
done | sort -t'|' -k5,5g
