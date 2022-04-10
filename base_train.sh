#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --output=log/base-train%j.out
#SBATCH --error=log/base-train%j.log
#SBATCH --gres=gpu:titan_xp:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID

set -o errexit
source activate tkg
srun python main.py \
  --config_path "configs/distmult.yml" \
  --data_dir "data/completion/" \
  --dataset "icews14" \
  --task "completion" \
  --do_test True \
  --batch_size 512 \
  --overwrite True \
  --resume_train False \
  --max_epochs 500 \
  --save_steps 40 \
  --eval_steps 20 \
  --neg_ratio 500 \
  --verbose False \
  --checkpoint "DistMult_0106-2021_500_icews14.pth"

echo finished at: `date`
exit 0;
~
