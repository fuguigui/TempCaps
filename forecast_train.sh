#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --output=log/dyrmlp-train%j.out
#SBATCH --error=log/dyrmlp-train%j.log
#SBATCH --gres=gpu:titan_xp:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=30G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID

set -o errexit
source activate tkg
srun python main.py \
  --config_path "configs/dyrmlp.yml" \
  --data_dir "data/forecast/" \
  --dataset "WIKI" \
  --task "forecasting" \
  --max_time_range 7 \
  --do_test True \
  --batch_size 100 \
  --overwrite True \
  --resume_train False \
  --max_epochs 40 \
  --save_steps 5 \
  --eval_steps 10 \
  --neg_ratio 0 \
  --test_size 256 \
  --from_pretrained False \
  --fix_pretrained False \
  --verbose False \
  --checkpoint "DyRMLP_0106-1106_60_ICEWS18.pth"


echo finished at: `date`
exit 0;
~
