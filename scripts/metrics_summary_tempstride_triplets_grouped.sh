#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/home/yzhidkova/logs"

patterns=(
  # many_c48_g15 original stride sweeps
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts1_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts4_fix_50ep-*.out"
  "mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts1_fix_50ep-*.out"
  "mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts2_fix_50ep-*.out"
  "mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts4_fix_50ep-*.out"
  "mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts1_fix_50ep-*.out"
  "mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts2_fix_50ep-*.out"
  "mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_ts4_fix_50ep-*.out"

  # many_c48_g15 fixedbins
  "mb_t0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts1_fix_50ep-*.out"
  "mb_t0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts2_fix_50ep-*.out"
  "mb_t0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts4_fix_50ep-*.out"
  "mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts1_fix_50ep-*.out"
  "mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts2_fix_50ep-*.out"
  "mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts4_fix_50ep-*.out"
  "mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts1_fix_50ep-*.out"
  "mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts2_fix_50ep-*.out"
  "mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts4_fix_50ep-*.out"

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

  # kanhead rbf residual
  "mb_kanhead_rbf_resid_t0_many_c48_g15_ts1_fix_50ep-*.out"
  "mb_kanhead_rbf_resid_t0_many_c48_g15_ts2_fix_50ep-*.out"
  "mb_kanhead_rbf_resid_t0_many_c48_g15_ts4_fix_50ep-*.out"
  "mb_kanhead_rbf_resid_s0_many_c48_g15_ts1_fix_50ep-*.out"
  "mb_kanhead_rbf_resid_s0_many_c48_g15_ts2_fix_50ep-*.out"
  "mb_kanhead_rbf_resid_s0_many_c48_g15_ts4_fix_50ep-*.out"
  "mb_kanhead_rbf_resid_uv0_many_c48_g15_ts1_fix_50ep-*.out"
  "mb_kanhead_rbf_resid_uv0_many_c48_g15_ts2_fix_50ep-*.out"
  "mb_kanhead_rbf_resid_uv0_many_c48_g15_ts4_fix_50ep-*.out"

  # kanhead spline residual
  "mb_kanhead_spline_resid_t0_ts1_fix_50ep-*.out"
  "mb_kanhead_spline_resid_t0_ts2_fix_50ep-*.out"
  "mb_kanhead_spline_resid_t0_ts4_fix_50ep-*.out"
  "mb_kanhead_spline_resid_s0_ts1_fix_50ep-*.out"
  "mb_kanhead_spline_resid_s0_ts2_fix_50ep-*.out"
  "mb_kanhead_spline_resid_s0_ts4_fix_50ep-*.out"
  "mb_kanhead_spline_resid_uv0_ts1_fix_50ep-*.out"
  "mb_kanhead_spline_resid_uv0_ts2_fix_50ep-*.out"
  "mb_kanhead_spline_resid_uv0_ts4_fix_50ep-*.out"

  # sdeenergy nll bins
  "mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts1_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts4_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts1_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts2_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts4_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_uv_sdeenergy_nll_bins_ts1_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_uv_sdeenergy_nll_bins_ts2_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_uv_sdeenergy_nll_bins_ts4_fix_50ep-*.out"

  # lambda sweep
  "mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l1e5_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l2e5_fix_50ep-*.out"
  "mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l1e4_fix_50ep-*.out"
)

TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

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
    # many_c48_g15 original
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

    # many_c48_g15 fixedbins
    mb_t0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts1_fix_50ep-*.out)
      short_name="many_c48_g15_fixedbins"; var="t0"; ts="1" ;;
    mb_t0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts2_fix_50ep-*.out)
      short_name="many_c48_g15_fixedbins"; var="t0"; ts="2" ;;
    mb_t0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts4_fix_50ep-*.out)
      short_name="many_c48_g15_fixedbins"; var="t0"; ts="4" ;;
    mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts1_fix_50ep-*.out)
      short_name="many_c48_g15_fixedbins"; var="s0"; ts="1" ;;
    mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts2_fix_50ep-*.out)
      short_name="many_c48_g15_fixedbins"; var="s0"; ts="2" ;;
    mb_s0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts4_fix_50ep-*.out)
      short_name="many_c48_g15_fixedbins"; var="s0"; ts="4" ;;
    mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts1_fix_50ep-*.out)
      short_name="many_c48_g15_fixedbins"; var="uv0"; ts="1" ;;
    mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts2_fix_50ep-*.out)
      short_name="many_c48_g15_fixedbins"; var="uv0"; ts="2" ;;
    mb_uv0_sdelight_kanhead_rbf_residual_sdeloss_bins_many_c48_g15_fixedbins_ts4_fix_50ep-*.out)
      short_name="many_c48_g15_fixedbins"; var="uv0"; ts="4" ;;

    # baseline
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

    # kanhead rbf residual
    mb_kanhead_rbf_resid_t0_many_c48_g15_ts1_fix_50ep-*.out)
      short_name="kanhead_rbf_resid"; var="t0"; ts="1" ;;
    mb_kanhead_rbf_resid_t0_many_c48_g15_ts2_fix_50ep-*.out)
      short_name="kanhead_rbf_resid"; var="t0"; ts="2" ;;
    mb_kanhead_rbf_resid_t0_many_c48_g15_ts4_fix_50ep-*.out)
      short_name="kanhead_rbf_resid"; var="t0"; ts="4" ;;
    mb_kanhead_rbf_resid_s0_many_c48_g15_ts1_fix_50ep-*.out)
      short_name="kanhead_rbf_resid"; var="s0"; ts="1" ;;
    mb_kanhead_rbf_resid_s0_many_c48_g15_ts2_fix_50ep-*.out)
      short_name="kanhead_rbf_resid"; var="s0"; ts="2" ;;
    mb_kanhead_rbf_resid_s0_many_c48_g15_ts4_fix_50ep-*.out)
      short_name="kanhead_rbf_resid"; var="s0"; ts="4" ;;
    mb_kanhead_rbf_resid_uv0_many_c48_g15_ts1_fix_50ep-*.out)
      short_name="kanhead_rbf_resid_uv"; var="uv0"; ts="1" ;;
    mb_kanhead_rbf_resid_uv0_many_c48_g15_ts2_fix_50ep-*.out)
      short_name="kanhead_rbf_resid_uv"; var="uv0"; ts="2" ;;
    mb_kanhead_rbf_resid_uv0_many_c48_g15_ts4_fix_50ep-*.out)
      short_name="kanhead_rbf_resid_uv"; var="uv0"; ts="4" ;;

    # kanhead spline residual
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
    mb_kanhead_spline_resid_uv0_ts1_fix_50ep-*.out)
      short_name="kanhead_spline_resid_uv"; var="uv0"; ts="1" ;;
    mb_kanhead_spline_resid_uv0_ts2_fix_50ep-*.out)
      short_name="kanhead_spline_resid_uv"; var="uv0"; ts="2" ;;
    mb_kanhead_spline_resid_uv0_ts4_fix_50ep-*.out)
      short_name="kanhead_spline_resid_uv"; var="uv0"; ts="4" ;;

    # sdeenergy nll bins
    mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts1_fix_50ep-*.out)
      short_name="sdeenergy_nll_bins"; var="t0"; ts="1" ;;
    mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_fix_50ep-*.out)
      short_name="sdeenergy_nll_bins"; var="t0"; ts="2" ;;
    mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts4_fix_50ep-*.out)
      short_name="sdeenergy_nll_bins"; var="t0"; ts="4" ;;
    mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts1_fix_50ep-*.out)
      short_name="sdeenergy_nll_bins"; var="s0"; ts="1" ;;
    mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts2_fix_50ep-*.out)
      short_name="sdeenergy_nll_bins"; var="s0"; ts="2" ;;
    mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_s0_ts4_fix_50ep-*.out)
      short_name="sdeenergy_nll_bins"; var="s0"; ts="4" ;;
    mb_sdelight_kanhead_rbf_residual_uv_sdeenergy_nll_bins_ts1_fix_50ep-*.out)
      short_name="sdeenergy_nll_bins_uv"; var="uv0"; ts="1" ;;
    mb_sdelight_kanhead_rbf_residual_uv_sdeenergy_nll_bins_ts2_fix_50ep-*.out)
      short_name="sdeenergy_nll_bins_uv"; var="uv0"; ts="2" ;;
    mb_sdelight_kanhead_rbf_residual_uv_sdeenergy_nll_bins_ts4_fix_50ep-*.out)
      short_name="sdeenergy_nll_bins_uv"; var="uv0"; ts="4" ;;

    # lambda sweep
    mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l1e5_fix_50ep-*.out)
      short_name="sdeenergy_nll_l1e5"; var="t0"; ts="2" ;;
    mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l2e5_fix_50ep-*.out)
      short_name="sdeenergy_nll_l2e5"; var="t0"; ts="2" ;;
    mb_sdelight_kanhead_rbf_residual_sdeenergy_nll_bins_t0_ts2_l1e4_fix_50ep-*.out)
      short_name="sdeenergy_nll_l1e4"; var="t0"; ts="2" ;;
  esac

  epoch_line="$(extract_last_match 'Epoch: [0-9]+' "$latest_file")"
  metric_line="$(extract_last_match 'mse:[0-9eE+.-]+, mae:[0-9eE+.-]+, rmse:[0-9eE+.-]+' "$latest_file")"

  ep="$(echo "$epoch_line" | sed -n 's/.*Epoch: \([0-9]\+\).*/\1/p')"
  train_loss="$(echo "$epoch_line" | sed -n 's/.*Train Loss: \([0-9.eE+-]\+\).*/\1/p')"
  vali_loss="$(echo "$epoch_line" | sed -n 's/.*Vali Loss: \([0-9.eE+-]\+\).*/\1/p')"

  mse="$(echo "$metric_line" | sed -n 's/.*mse:\([0-9.eE+-]\+\), mae:.*/\1/p')"
  mae="$(echo "$metric_line" | sed -n 's/.*mae:\([0-9.eE+-]\+\), rmse:.*/\1/p')"
  rmse="$(echo "$metric_line" | sed -n 's/.*rmse:\([0-9.eE+-]\+\).*/\1/p')"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${var}" "${short_name}" "${ts}" "${ep:-?}" "${train_loss:-?}" "${vali_loss:-?}" "${mse:-?}" "${mae:-?}" "${rmse:-?}" "${base_name}" >> "$TMP_FILE"
done

print_group() {
  local group="$1"
  local title="$2"

  local rows
  rows="$(awk -F'\t' -v g="$group" '$1==g' "$TMP_FILE" | sort -t$'\t' -k7,7g || true)"

  [[ -z "$rows" ]] && return

  echo
  echo "=== ${title} ==="
  printf "%-30s | %-3s | %-6s | %-12s | %-12s | %-12s | %-12s | %-12s | %s\n" \
    "model" "ts" "ep" "train_loss" "vali_loss" "mse" "mae" "rmse" "file"
  printf '%s\n' "----------------------------------------------------------------------------------------------------------------------------------------------------------------"

  while IFS=$'\t' read -r var short_name ts ep train_loss vali_loss mse mae rmse base_name; do
    printf "%-30s | %-3s | %-6s | %-12s | %-12s | %-12s | %-12s | %-12s | %s\n" \
      "${short_name}" "${ts}" "${ep}" "${train_loss}" "${vali_loss}" "${mse}" "${mae}" "${rmse}" "${base_name}"
  done <<< "$rows"
}

print_group "t0" "t0"
print_group "s0" "s0"
print_group "uv0" "uv0"
