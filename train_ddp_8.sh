#!/bin/bash
#SBATCH -A EUHPC_D32_008
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00
#SBATCH --job-name=cinema_8gpu
#SBATCH --output=ddp_8gpu_%j.log
#SBATCH --nodes=2              # <---【修改1】申请 2 个物理节点
#SBATCH --ntasks-per-node=1    # <--- 每个节点启动 1 个 torchrun 进程 (总共启动 2 个)
#SBATCH --gres=gpu:4           # <---【注意】保持 4！这是指“每个节点”要 4 张卡
#SBATCH --cpus-per-task=32     # <--- 保持 32 (每个节点配 32 核)

# 1. 加载环境
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0

# 2. 激活虚拟环境
source $CINECA_SCRATCH/envs/torch/bin/activate

# 【重要防报错】强制进入代码目录
# 请确保这个路径是你存放 run_ddp.py 的真实路径
cd $CINECA_SCRATCH/Cinema  
echo "Working in directory: $(pwd)"

# 3. DDP 网络配置
# 获取主节点 IP
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Master Node: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: 8 GPUs (2 Nodes x 4 GPUs)"

# 4. 启动命令
# srun 会在 2 个节点上各运行一次 python 命令，组成分布式集群
srun python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    run_ddp.py --config_data mra_atlas