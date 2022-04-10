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
  --dataset "gdelt" \
  --checkpoint "DyRMLP_0206-2144_14_gdelt.pth" \
  --do_train False \
  --do_test True \
  --max_time_range 3 \
  --overwrite False \
  --test_size 256 \
  --save_eval True


echo finished at: `date`
exit 0;
~
