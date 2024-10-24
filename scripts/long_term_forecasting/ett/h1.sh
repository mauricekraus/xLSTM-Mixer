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

seeds=(2021 2022 2023)
dataset="ETTh1"

pred_len=96
seq_len=512
slstm_num_heads=8
slstm_conv1d_kernel_size=0
xlstm_embedding_dim=256
xlstm_num_blocks=1
init_token=2
xlstm_dropout=0.25
lr=0.001
batch_size=32
gamma=0.98
cosine_epochs=15
warmup_epochs=5
constant_gamma_epochs=2

for seed in "${seeds[@]}"; do
    python -m xlstm_mixer fit+test \
        --data ForecastingTSLibDataModule \
        --data.dataset_name $dataset \
        --optimizer.lr $lr \
        --data.seq_len $seq_len \
        --data.pred_len $pred_len \
        --data.label_len 0 \
        --data.batch_size $batch_size \
        --data.num_workers 4 \
        --data.persistent_workers true \
        --model LongTermForecastingExp \
        --model.criterion torch.nn.L1Loss \
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
        --trainer.max_epochs 40 \
        --seed_everything $seed \
        --trainer.fast_dev_run $dev_run
done

pred_len=192
seq_len=512
lr=0.0002
batch_size=16
init_token=2
xlstm_embedding_dim=1024
slstm_num_heads=32
slstm_conv1d_kernel_size=4
xlstm_num_blocks=4
xlstm_dropout=0.1
warmup_epochs=5 
cosine_epochs=20
constant_gamma=0.99
constant_gamma_epochs= 1 
for seed in "${seeds[@]}"; do
    python -m xlstm_mixer fit+test \
        --data ForecastingTSLibDataModule \
        --data.dataset_name $dataset \
        --optimizer.lr $lr \
        --data.seq_len $seq_len \
        --data.pred_len $pred_len \
        --data.label_len 0 \
        --data.batch_size $batch_size \
        --data.num_workers 4 \
        --data.persistent_workers true \
        --model LongTermForecastingExp \
        --model.criterion torch.nn.L1Loss \
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
        --trainer.max_epochs 40 \
        --seed_everything $seed \
        --trainer.fast_dev_run $dev_run
done

pred_len=336
seq_len=512
lr=0.0002
batch_size=64
init_token=3
xlstm_embedding_dim=128
slstm_num_heads=4
slstm_conv1d_kernel_size=4
xlstm_dropout=0.1
warmup_epochs=5 
cosine_epochs=20
constant_gamma=0.99
constant_gamma_epochs= 1 
for seed in "${seeds[@]}"; do
    python -m xlstm_mixer fit+test \
        --data ForecastingTSLibDataModule \
        --data.dataset_name $dataset \
        --optimizer.lr $lr \
        --data.seq_len $seq_len \
        --data.pred_len $pred_len \
        --data.label_len 0 \
        --data.batch_size $batch_size \
        --data.num_workers 4 \
        --data.persistent_workers true \
        --model LongTermForecastingExp \
        --model.criterion torch.nn.L1Loss \
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
        --trainer.max_epochs 40 \
        --seed_everything $seed \
        --trainer.fast_dev_run $dev_run
done

pred_len=720
seq_len=768
slstm_num_heads=8
slstm_conv1d_kernel_size=0
xlstm_embedding_dim=128
xlstm_num_blocks=1
init_token=1
xlstm_dropout=0.25
lr=0.001
batch_size=32
gamma=0.98
cosine_epochs=15
warmup_epochs=5
constant_gamma_epochs=2
for seed in "${seeds[@]}"; do
    python -m xlstm_mixer fit+test \
        --data ForecastingTSLibDataModule \
        --data.dataset_name $dataset \
        --optimizer.lr $lr \
        --data.seq_len $seq_len \
        --data.pred_len $pred_len \
        --data.label_len 0 \
        --data.batch_size $batch_size \
        --data.num_workers 4 \
        --data.persistent_workers true \
        --model LongTermForecastingExp \
        --model.criterion torch.nn.L1Loss \
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
        --trainer.max_epochs 40 \
        --seed_everything $seed \
        --trainer.fast_dev_run $dev_run
done