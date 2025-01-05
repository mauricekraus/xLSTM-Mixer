import argparse
import subprocess
import os
# Default values
dev_run = False

# Define the function to run the experiment
def run_experiment(seed, dev_run):
    dataset = "ETTm1"
    pred_len = 96
    seq_len = 768
    xlstm_dropout = 0.1
    lr = 0.001
    batch_size = 4
    init_token = 2
    xlstm_embedding_dim = 256
    slstm_num_heads = 8
    slstm_conv1d_kernel_size = 0
    xlstm_num_blocks = 1
    gamma = 0.98
    cosine_epochs = 15
    warmup_epochs = 5
    constant_gamma_epochs = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    # Prepare the command arguments
    command = [
        "python", "-m", "xlstm_mixer", "fit+test",
        "--data", "ForecastingTSLibDataModule",
        "--data.dataset_name", dataset,
        "--optimizer.lr", str(lr),
        "--data.seq_len", str(seq_len),
        "--data.pred_len", str(pred_len),
        "--data.label_len", "0",
        "--data.batch_size", str(batch_size),
        "--data.num_workers", "4",
        "--data.persistent_workers", "true",
        "--model", "LongTermForecastingExp",
        "--model.criterion", "torch.nn.L1Loss",
        "--model.architecture", "xLSTMMixer",
        "--model.architecture.num_mem_tokens", str(init_token),
        "--model.architecture.xlstm_num_heads", str(slstm_num_heads),
        "--model.architecture.xlstm_num_blocks", str(xlstm_num_blocks),
        "--model.architecture.xlstm_embedding_dim", str(xlstm_embedding_dim),
        "--model.architecture.xlstm_conv1d_kernel_size", str(slstm_conv1d_kernel_size),
        "--model.architecture.xlstm_dropout", str(xlstm_dropout),
        "--optimizer.lr", str(lr),
        "--lr_scheduler.constant_gamma_epochs", str(constant_gamma_epochs),
        "--lr_scheduler.gamma", str(gamma),
        "--lr_scheduler.cosine_epochs", str(cosine_epochs),
        "--lr_scheduler.warmup_epochs", str(warmup_epochs),
        "--trainer.logger.name", f"{dataset}_xlstm-mixer_{pred_len}_{seed}",
        "--trainer.logger.project", "xlstm-mixer",
        "--trainer.max_epochs", "40",
        "--seed_everything", str(seed),
        "--trainer.fast_dev_run", str(dev_run).lower()  # Convert boolean to string
    ]

    # Execute the command
    subprocess.run(command)

# Argument parsing for --dev flag
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run xLSTM Mixer experiments")
    parser.add_argument(
        "--dev", action="store_true", help="Run in development (debug) mode"
    )
    return parser.parse_args()

# Main entry point
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # If --dev flag is passed, set dev_run to True
    dev_run = args.dev

    # List of seeds to loop over
    seeds = [2021, 2022, 2023]

    # Loop through the seeds and run the experiment for each
    for seed in seeds:
        run_experiment(seed, dev_run)
