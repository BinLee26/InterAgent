HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 2025 train.py \
    --config-name=train.yaml \
    hydra.run.dir=/data/logs \

