#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/home/yzhidkova/logs"

patterns=(
  # many_c48_g15 stride sweeps
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts1_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts4_fix_50ep-*.out"
  "mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts1_fix_50ep-*.out"
  "mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts2_fix_50ep-*.out"
  "mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts4_fix_50ep-*.out"
  "mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts1_fix_50ep-*.out"
  "mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts2_fix_50ep-*.out"
  "mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts4_fix_50ep-*.out"

  # baseline stride sweeps
  "mb_baseline_t0_ts1_fix_50ep-*.out"
  "mb_baseline_t0_ts2_fix_50ep-*.out"
  "mb_baseline_t0_ts4_fix_50ep-*.out"
  "mb_baseline_s0_ts1_fix_50ep-*.out"
  "mb_baseline_s0_ts2_fix_50ep-*.out"
  "mb_baseline_s0_ts4_fix_50ep-*.out"
  "mb_baseline_uv0_ts1_fix_50ep-*.out"
  "mb_baseline_uv0_ts2_fix_50ep-*.out"
  "mb_baseline_uv0_ts4_fix_50ep-*.out"

  # kanhead rbf residual stride sweep
  "mb_kanhead_rbf_resid_t0_many_c48_g15_ts1_fix_50ep-*.out"
  "mb_kanhead_rbf_resid_t0_many_c48_g15_ts2_fix_50ep-*.out"
  "mb_kanhead_rbf_resid_t0_many_c48_g15_ts4_fix_50ep-*.out"

  # kanhead spline residual stride sweeps
  "mb_kanhead_spline_resid_t0_ts1_fix_50ep-*.out"
  "mb_kanhead_spline_resid_t0_ts2_fix_50ep-*.out"
  "mb_kanhead_spline_resid_t0_ts4_fix_50ep-*.out"
  "mb_kanhead_spline_resid_s0_ts1_fix_50ep-*.out"
  "mb_kanhead_spline_resid_s0_ts2_fix_50ep-*.out"
  "mb_kanhead_spline_resid_s0_ts4_fix_50ep-*.out"
)

printf "%-38s | %-4s | %-3s | %-6s | %-12s | %-12s | %-12s | %-12s | %-12s | %s\n" \
  "model" "var" "ts" "ep" "train_loss" "vali_loss" "mse" "mae" "rmse" "file"
printf '%s\n' "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

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

  latest_file="$(ls -1t "${files[@]}" | head -n 1)"
  base_name="$(basename "$latest_file")"

  short_name="$base_name"
  var="?"
  ts="?"

  case "$base_name" in
    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts1_fix_50ep-*.out)
      short_name="many_c48_g15"; var="t0"; ts="1" ;;
    mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts4_fix_50ep-*.out)
      short_name="many_c48_g15"; var="t0"; ts="4" ;;

    mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts1_fix_50ep-*.out)
      short_name="many_c48_g15"; var="s0"; ts="1" ;;
    mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts2_fix_50ep-*.out)
      short_name="many_c48_g15"; var="s0"; ts="2" ;;
    mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts4_fix_50ep-*.out)
      short_name="many_c48_g15"; var="s0"; ts="4" ;;

    mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts1_fix_50ep-*.out)
      short_name="many_c48_g15"; var="uv0"; ts="1" ;;
    mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts2_fix_50ep-*.out)
      short_name="many_c48_g15"; var="uv0"; ts="2" ;;
    mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts4_fix_50ep-*.out)
      short_name="many_c48_g15"; var="uv0"; ts="4" ;;

    mb_baseline_t0_ts1_fix_50ep-*.out)
      short_name="baseline"; var="t0"; ts="1" ;;
    mb_baseline_t0_ts2_fix_50ep-*.out)
      short_name="baseline"; var="t0"; ts="2" ;;
    mb_baseline_t0_ts4_fix_50ep-*.out)
      short_name="baseline"; var="t0"; ts="4" ;;

    mb_baseline_s0_ts1_fix_50ep-*.out)
      short_name="baseline"; var="s0"; ts="1" ;;
    mb_baseline_s0_ts2_fix_50ep-*.out)
      short_name="baseline"; var="s0"; ts="2" ;;
    mb_baseline_s0_ts4_fix_50ep-*.out)
      short_name="baseline"; var="s0"; ts="4" ;;

    mb_baseline_uv0_ts1_fix_50ep-*.out)
      short_name="baseline_uv"; var="uv0"; ts="1" ;;
    mb_baseline_uv0_ts2_fix_50ep-*.out)
      short_name="baseline_uv"; var="uv0"; ts="2" ;;
    mb_baseline_uv0_ts4_fix_50ep-*.out)
      short_name="baseline_uv"; var="uv0"; ts="4" ;;

    mb_kanhead_rbf_resid_t0_many_c48_g15_ts1_fix_50ep-*.out)
      short_name="kanhead_rbf_resid"; var="t0"; ts="1" ;;
    mb_kanhead_rbf_resid_t0_many_c48_g15_ts2_fix_50ep-*.out)
      short_name="kanhead_rbf_resid"; var="t0"; ts="2" ;;
    mb_kanhead_rbf_resid_t0_many_c48_g15_ts4_fix_50ep-*.out)
      short_name="kanhead_rbf_resid"; var="t0"; ts="4" ;;

    mb_kanhead_spline_resid_t0_ts1_fix_50ep-*.out)
      short_name="kanhead_spline_resid"; var="t0"; ts="1" ;;
    mb_kanhead_spline_resid_t0_ts2_fix_50ep-*.out)
      short_name="kanhead_spline_resid"; var="t0"; ts="2" ;;
    mb_kanhead_spline_resid_t0_ts4_fix_50ep-*.out)
      short_name="kanhead_spline_resid"; var="t0"; ts="4" ;;

    mb_kanhead_spline_resid_s0_ts1_fix_50ep-*.out)
      short_name="kanhead_spline_resid"; var="s0"; ts="1" ;;
    mb_kanhead_spline_resid_s0_ts2_fix_50ep-*.out)
      short_name="kanhead_spline_resid"; var="s0"; ts="2" ;;
    mb_kanhead_spline_resid_s0_ts4_fix_50ep-*.out)
      short_name="kanhead_spline_resid"; var="s0"; ts="4" ;;
  esac

  epoch_line="$(extract_last_match 'Epoch: [0-9]+' "$latest_file")"
  metric_line="$(extract_last_match 'mse:[0-9eE+.-]+, mae:[0-9eE+.-]+, rmse:[0-9eE+.-]+' "$latest_file")"

  ep="$(echo "$epoch_line" | sed -n 's/.*Epoch: \([0-9]\+\).*/\1/p')"
  train_loss="$(echo "$epoch_line" | sed -n 's/.*Train Loss: \([0-9.eE+-]\+\).*/\1/p')"
  vali_loss="$(echo "$epoch_line" | sed -n 's/.*Vali Loss: \([0-9.eE+-]\+\).*/\1/p')"

  mse="$(echo "$metric_line" | sed -n 's/.*mse:\([0-9.eE+-]\+\), mae:.*/\1/p')"
  mae="$(echo "$metric_line" | sed -n 's/.*mae:\([0-9.eE+-]\+\), rmse:.*/\1/p')"
  rmse="$(echo "$metric_line" | sed -n 's/.*rmse:\([0-9.eE+-]\+\).*/\1/p')"

  printf "%-38s | %-4s | %-3s | %-6s | %-12s | %-12s | %-12s | %-12s | %-12s | %s\n" \
    "${short_name}" "${var}" "${ts}" "${ep:-?}" "${train_loss:-?}" "${vali_loss:-?}" "${mse:-?}" "${mae:-?}" "${rmse:-?}" "${base_name}"
done | sort -t'|' -k7,7g
