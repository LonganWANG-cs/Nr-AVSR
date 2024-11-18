#!/bin/bash

#SBATCH --job-name=train_1    # Job name
#SBATCH --output=train_output.txt # Standard output and error log
#SBATCH --error=train_error.txt  # Error log
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=32G                      # Request 16GB of memory
# Activate the environment if needed
# source activate MASR  # Replace 'torch' with the name of your conda environment

# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/e/e1344641/SMC/Nr-MASR/src
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/home/e/e1344641/SMC/Nr-MASR
cd $run_dir
code_dir=./mala_asr_slidespeech

speech_encoder_path=/home/e/e1344641/SMC/Nr-MASR/models/WavLM-Large.pt
llm_path=/home/e/e1344641/SMC/Nr-MASR/models/vicuna-7b-v1.5
output_dir=/home/e/e1344641/SMC/Nr-MASR/results/finetrun/finetune_MaLa-ASR_withkeywords_L95-$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=wavlm \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=cov1d-linear \
++dataset_config.dataset=slidespeech_dataset \
++dataset_config.input_type=raw \
++dataset_config.use_ocr=true \
++train_config.model_name=mala_asr \
++train_config.num_epochs=5 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=110000 \
++train_config.lr=5e-5 \
++train_config.validation_interval=2000 \
++train_config.batch_size_training=6 \
++train_config.val_batch_size=6 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

# # -m debugpy --listen 5678 --wait-for-client
# if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
#     python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_mala_asr.py \
#         --config-path "conf" \
#         --config-name "prompt.yaml" \
#         $hydra_args
# else
#     torchrun \
#         --nnodes 1 \
#         --nproc_per_node 4 \
#         --master_port=29503 \
#         $code_dir/finetune_mala_asr.py \
#         --config-path "conf" \
#         --config-name "prompt.yaml" \
#         ++train_config.enable_fsdp=false \
#         ++train_config.enable_ddp=true \
#         ++train_config.use_fp16=true \
#         $hydra_args
# fi
# srun python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_mala_asr.py \
#         --config-path "conf" \
#         --config-name "prompt.yaml" \
#         $hydra_args
srun python $code_dir/finetune_mala_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args