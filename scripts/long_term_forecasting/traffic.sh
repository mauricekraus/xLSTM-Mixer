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
dataset="Traffic"

pred_len=96
seq_len=336
lr=0.0005
batch_size=32
init_token=3
xlstm_embedding_dim=1024
slstm_conv1d_kernel_size=4
xlstm_num_blocks=2
slstm_num_heads=16
xlstm_dropout=0.1
gamma=0.99
cosine_epochs=20
warmup_epochs=5
constant_gamma_epochs=1

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
seq_len=336
lr=0.0005
batch_size=16
init_token=2
xlstm_embedding_dim=1024
slstm_conv1d_kernel_size=4
xlstm_num_blocks=4
slstm_num_heads=32
xlstm_dropout=0.1
gamma=0.99
cosine_epochs=20
warmup_epochs=5
constant_gamma_epochs=1

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
seq_len=336
lr=0.0005
batch_size=16
init_token=2
xlstm_embedding_dim=1024
slstm_conv1d_kernel_size=4
xlstm_num_blocks=4
slstm_num_heads=32
xlstm_dropout=0.1
gamma=0.99
cosine_epochs=20
warmup_epochs=5
constant_gamma_epochs=1

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
seq_len=336
lr=0.0005
batch_size=32
init_token=2
xlstm_embedding_dim=1024
slstm_conv1d_kernel_size=4
xlstm_num_blocks=4
slstm_num_heads=16
xlstm_dropout=0.1
gamma=0.99
cosine_epochs=20
warmup_epochs=5
constant_gamma_epochs=1

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