#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --output=log/base-eval%j.out
#SBATCH --error=log/base-eval%j.log
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
  --do_train False \
  --test_size 256 \
  --overwrite False \
  --resume_train False \
  --checkpoint "DistMult_3105-0852_40_icews14.pth"
  --neg_ratio 500 \
  --verbose False

echo finished at: `date`
exit 0;
~
