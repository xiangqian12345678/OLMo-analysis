# Pycharm调试训练

## Parameters

    --save-folder=output/190m/pretrain
    --name=olmo3-190m-pretrain
    --sequence-length=1024

## Working directory

    项目根目录

## 环境变量

    Environment variables中添加：
    CUDA_VISIBLE_DEVICES=0
    MASTER_ADDR=localhost
    MASTER_PORT=29500
    WORLD_SIZE=1
    RANK=0
    LOCAL_RANK=0
    LOCAL_WORLD_SIZE=1
    NUM_NODES=1
    OLMO_SHARED_FS=1

CUDA_VISIBLE_DEVICES=0;MASTER_ADDR=localhost;MASTER_PORT=29500;WORLD_SIZE=1;RANK=0;LOCAL_RANK=0;LOCAL_WORLD_SIZE=1;NUM_NODES=1;OLMO_SHARED_FS=1