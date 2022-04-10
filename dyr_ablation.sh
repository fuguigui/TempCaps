#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --output=log/dyrmlp-abalation%j.out
#SBATCH --error=log/dyrmlp-abalation%j.log
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
  --dataset "icews14" \
  --max_time_range 3 \
  --task "completion" \
  --do_test True \
  --batch_size 300 \
  --overwrite False \
  --resume_train False \
  --max_epochs 240 \
  --save_steps 80 \
  --eval_steps -1 \
  --neg_ratio 0 \
  --test_size 256 \
  --from_pretrained False \
  --fix_pretrained False \
  --verbose False


echo finished at: `date`
exit 0;
~
