#!/bin/bash
#SBATCH --account=aip-papernot
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/project/6105016/shayuxin/verifying-robust-networks/LBDN/grid_logs/stdout-%j.log
#SBATCH --error=/project/6105016/shayuxin/verifying-robust-networks/LBDN/grid_logs/stderr-%j.log

# Train MLP with a given Lipschitz training method and loss over a depth x width grid
# --resume continues previous training when existing up to EPOCHS total

# Usage: sbatch sbatch_toy_grid.sh <layer> [loss] [batchnorm] [margin]
#   batchnorm: 1 to add BatchNorm after each hidden layer (default 0)
#   margin:    multimargin loss margin (default 0.5; ignored for xent)
#   e.g. sbatch sbatch_toy_grid.sh Sandwich multimargin
#   e.g. sbatch sbatch_toy_grid.sh Plain xent
#   e.g. sbatch sbatch_toy_grid.sh Sandwich multimargin 1 1.0
set -e

LAYER="$1"
LOSS="${2:-xent}"
BATCHNORM="${3:-0}"
MARGIN="${4:-0.5}"
VALID_LAYERS=(Plain Sandwich Orthogon Aol SLL)
VALID_LOSSES=(xent multimargin)
if [[ ! " ${VALID_LAYERS[*]} " =~ " ${LAYER} " ]]; then
  echo "Usage: sbatch sbatch_toy_grid.sh <layer> [loss], where <layer> is one of: ${VALID_LAYERS[*]}"
  exit 1
fi
if [[ ! " ${VALID_LOSSES[*]} " =~ " ${LOSS} " ]]; then
  echo "Usage: sbatch sbatch_toy_grid.sh <layer> [loss], where [loss] is one of: ${VALID_LOSSES[*]}"
  exit 1
fi

EXTRA_ARGS=()
[[ "$BATCHNORM" == "1" ]] && EXTRA_ARGS+=(--batchnorm)
[[ "$LOSS" == "multimargin" ]] && EXTRA_ARGS+=(--margin "$MARGIN")
TAG="${LAYER}_${LOSS}"
[[ "$LOSS" == "multimargin" && "$MARGIN" != "0.5" ]] && TAG+="_margin${MARGIN}"
[[ "$BATCHNORM" == "1" ]] && TAG+="_bn"

REPO_ROOT="/project/6105016/shayuxin/verifying-robust-networks/LBDN"
cd "$REPO_ROOT"

source /project/6105016/shayuxin/verifying-robust-networks/.venv/bin/activate

DATASET="mnist"
EPOCHS=100   # training run to 100 epochs
GAMMA=1
SEED=123
SAVE_FREQ=20   # keep a permanent checkpoint snapshot every N epochs

DEPTHS=(2 4 8)
WIDTHS=(128 256 512)

RESULTS_DIR="$REPO_ROOT/grid_logs"
mkdir -p "$RESULTS_DIR"
SUMMARY_CSV="$RESULTS_DIR/results_${TAG}_${SLURM_JOB_ID:-manual}.csv"
echo "layer,loss,width,depth,train_acc,test_acc,logfile" > "$SUMMARY_CSV"

for DEPTH in "${DEPTHS[@]}"; do
  for WIDTH in "${WIDTHS[@]}"; do
    if [[ "$WIDTH" == "512" && "$DEPTH" == "8" ]]; then
      echo "== skip w=512,d=8 =="
      continue
    fi

    RUN_LOG="$RESULTS_DIR/${TAG}_w${WIDTH}_d${DEPTH}.log"
    echo "=== Training layer=$LAYER loss=$LOSS bn=$BATCHNORM margin=$MARGIN width=$WIDTH depth=$DEPTH ==="

    python main.py --mode train --model Toy \
      --layer "$LAYER" --scale small --width "$WIDTH" --depth "$DEPTH" \
      --dataset "$DATASET" --gamma "$GAMMA" --seed "$SEED" --epochs "$EPOCHS" \
      --loss "$LOSS" --resume --save_freq "$SAVE_FREQ" "${EXTRA_ARGS[@]}" \
      2>&1 | tee "$RUN_LOG"

    LAST_ACC_LINE=$(grep 'acc:' "$RUN_LOG" | tail -n 1)
    ACC_FIELD=$(echo "$LAST_ACC_LINE" | grep -oP 'acc: \K[0-9.]+/[0-9.]+' || true)
    TRAIN_ACC=$(echo "$ACC_FIELD" | cut -d/ -f1)
    TEST_ACC=$(echo "$ACC_FIELD" | cut -d/ -f2)

    echo "$LAYER,$LOSS,$WIDTH,$DEPTH,${TRAIN_ACC:-NA},${TEST_ACC:-NA},$RUN_LOG" >> "$SUMMARY_CSV"
  done
done

echo "Grid sweep complete for layer=$LAYER loss=$LOSS. Summary written to $SUMMARY_CSV"
