#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --output=log/dyrmlp-test%j.out
#SBATCH --error=log/dyrmlp-test%j.log
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
  --config_path "configs/dyrmlp.yml" \
  --data_dir "data/completion/" \
  --task "completion" \
  --dataset "icews11-14" \
  --checkpoint "DyRMLP_0106-1428_90_icews11-14.pth" \
  --do_train False \
  --do_test True \
  --max_time_range 10 \
  --overwrite False \
  --test_size 256


echo finished at: `date`
exit 0;
~
