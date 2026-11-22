#!/usr/bin/env bash
# One-command jobs for Kaggle "Save & Run All" sessions (12 h limit).
#   bash scripts/kaggle_job.sh tiw    # data + discriminator + TIW-DSM training
#   bash scripts/kaggle_job.sh obs    # data + DSM(obs) training
#   bash scripts/kaggle_job.sh both   # T4 x2: DSM(obs) on GPU 1 and discriminator + TIW-DSM on GPU 0, in parallel
#   bash scripts/kaggle_job.sh eval   # needs the tiw and obs outputs attached as inputs
set -euo pipefail
JOB=${1:?job}
START=$(date +%s)
BUDGET_H=${BUDGET_H:-11.2}           # stop training early enough to save outputs
elapsed_h() { python -c "import time;print((time.time()-$START)/3600)"; }
left_h() { python -c "print(max(0.2, $BUDGET_H - $(elapsed_h)))"; }

pip -q install pytorch-fid==0.3.0 gdown pyarrow >/dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python scripts/prepare_data.py

case "$JOB" in
  tiw)
    python scripts/train_discriminator.py --iters "${DISC_ITERS:-10000}"
    python scripts/train_score.py --mode tiw --outdir runs/tiw --duration_mimg "${MIMG:-11}" --time_limit_h "$(left_h)"
    ;;
  both)
    CUDA_VISIBLE_DEVICES=1 python scripts/train_score.py --mode obs --outdir runs/obs --duration_mimg "${MIMG:-11}" \
        --time_limit_h "$(left_h)" > runs_obs.log 2>&1 &
    OBS_PID=$!
    export CUDA_VISIBLE_DEVICES=0
    python scripts/train_discriminator.py --iters "${DISC_ITERS:-10000}"
    python scripts/train_score.py --mode tiw --outdir runs/tiw --duration_mimg "${MIMG:-11}" --time_limit_h "$(left_h)"
    wait $OBS_PID || echo "obs exited with an error"; tail -5 runs_obs.log; mkdir -p runs; cp runs_obs.log runs/
    ;;
  obs)
    python scripts/train_score.py --mode obs --outdir runs/obs --duration_mimg "${MIMG:-11}" --time_limit_h "$(left_h)"
    ;;
  eval)
    # copy the attached training outputs into place
    for name in tiw obs; do
      src=$(find /kaggle/input -type d -path "*runs/$name" | head -1)
      mkdir -p runs && cp -r "$src" runs/
    done
    for f in discriminator_train.json discriminator_diag.json; do
      p=$(find /kaggle/input -name "$f" | head -1); [ -n "$p" ] && cp "$p" results/
    done
    # compare both models at the largest training length reached by BOTH (equal Mimg)
    K=$(python - <<'EOF'
import glob, os, re
ks = [set(int(re.search(r"ema_(\d+)k", f).group(1)) for f in glob.glob(f"runs/{n}/ema_*k.pt")) for n in ("tiw", "obs")]
print(f"{max(ks[0] & ks[1]):06d}")
EOF
)
    echo "evaluating both models at ${K} kimg"
    python scripts/evaluate.py --ckpt runs/obs/ema_${K}k.pt --name obs --num 50000
    python scripts/evaluate.py --ckpt runs/tiw/ema_${K}k.pt --name tiw --num 50000
    for r in bias ref target; do python scripts/evaluate.py --real $r --name data_$r; done
    python scripts/steps_experiment.py --ckpt obs=runs/obs/ema_${K}k.pt --ckpt tiw=runs/tiw/ema_${K}k.pt --num 10000
    python scripts/make_report_assets.py --runs obs=runs/obs tiw=runs/tiw
    mkdir -p checkpoints/deploy
    cp runs/tiw/ema_${K}k.pt checkpoints/deploy/tiw_dsm.pt
    cp runs/obs/ema_${K}k.pt checkpoints/deploy/dsm_obs.pt
    ;;
esac
rm -rf data/cifar10/*.parquet
echo "job $JOB done in $(elapsed_h) h"
