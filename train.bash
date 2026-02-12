# 1. 先进入脚本所在目录
cd /leonardo_scratch/large/userexternal/xzhang01/Cinema/

# 2. 赋予执行权限 (保险起见)
chmod +x train.bash

# 3. 使用 sbatch 提交 (核心修改在最后一行)
sbatch -A EUHPC_D32_008 \
       -p boost_usr_prod \
       --time 24:00:00 \
       --gres=gpu:1 \
       --cpus-per-task=20 \
       --job-name=cinema_train \
       --output=train_%j.log \
       --wrap="module purge; module load profile/deeplrn; module load cineca-ai/4.3.0; source $CINECA_SCRATCH/envs/torch/bin/activate; python run.py --config_data mra_atlas"