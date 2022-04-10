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
  --data_dir "data/completion/" \
  --dataset "icews14" \
  --task "completion" \
  --max_time_range 3 \
  --do_test True \
  --batch_size 300 \
  --overwrite False \
  --resume_train False \
  --max_epochs 240 \
  --save_steps 30 \
  --eval_steps 60 \
  --neg_ratio 0 \
  --test_size 256 \
  --from_pretrained False \
  --fix_pretrained False \
  --verbose False \
  --save_eval True \
  --checkpoint "DyRMLP_2305-1921_240_icews14.pth"


echo finished at: `date`
exit 0;
~
