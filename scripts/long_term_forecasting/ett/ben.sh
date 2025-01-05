#!/bin/bash
# Default: Not a dev run
dev_run=false

# Check if "--dev" flag is passed
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dev) dev_run=true ;;  # Set dev_run to true if --dev flag is present
    esac
    shift
done

seeds=(2023)
dataset="EthanolConcentration"

pred_len=96
seq_len=768
xlstm_dropout=0.1
lr=0.001
batch_size=32
init_token=2
xlstm_embedding_dim=256
slstm_num_heads=8
slstm_conv1d_kernel_size=0
xlstm_num_blocks=1
gamma=0.98
cosine_epochs=15
warmup_epochs=5
constant_gamma_epochs=2

export CUDA_VISIBLE_DEVICES=0

for seed in "${seeds[@]}"; do
    python -m xlstm_mixer test-classification \
        --data ClassificationTSLibDataModule \
        --data.dataset_name $dataset \
        --optimizer.lr $lr \
        --data.seq_len $seq_len \
        --data.num_workers 4 \
        --model ClassificationExp \
        --model.architecture xLSTMMixer  \
        --model.architecture.num_mem_tokens $init_token \
        --model.architecture.xlstm_num_heads $slstm_num_heads \
        --model.architecture.xlstm_num_blocks $xlstm_num_blocks \
        --model.architecture.xlstm_embedding_dim $xlstm_embedding_dim \
        --model.architecture.xlstm_conv1d_kernel_size $slstm_conv1d_kernel_size \
        --model.architecture.xlstm_dropout $xlstm_dropout \
        --optimizer.lr $lr \
        --lr_scheduler.constant_gamma_epochs $constant_gamma_epochs \
        --lr_scheduler.gamma $gamma \
        --lr_scheduler.cosine_epochs $cosine_epochs \
        --lr_scheduler.warmup_epochs $warmup_epochs \
        --trainer.logger.name "${dataset}_xlstm-mixer_${pred_len}_$seed" \
        --trainer.logger.project xlstm-mixer \
        --trainer.max_epochs 2 \
        --seed_everything $seed \
        --trainer.fast_dev_run $dev_run
done

