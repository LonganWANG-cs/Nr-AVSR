#!/bin/bash

#SBATCH --job-name=test_1    # Job name
#SBATCH --output=videogs_loss_output.txt # Standard output and error log
#SBATCH --error=videogs_loss_error.txt  # Error log
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100-40:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=32G                      # Request 16GB of memory
# Activate the environment if needed
source activate MASR  # Replace 'torch' with the name of your conda environment

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=$PYTHONPATH:/home/e/e1344641/SMC/Nr-MASR/src
run_dir=/home/e/e1344641/SMC/Nr-MASR
cd $run_dir
code_dir=./mala_asr_slidespeech

speech_encoder_path=/home/e/e1344641/SMC/Nr-MASR/models/WavLM-Large.pt
llm_path=/home/e/e1344641/SMC/Nr-MASR/models/vicuna-7b-v1.5

output_dir=/home/e/e1344641/SMC/Nr-MASR/results
ckpt_path=$output_dir/asr
split=dev
# val_data_path=/home/e/e1344641/SMC/Nr-MASR/data/${split}_oracle_v1/
val_data_path=/home/e/e1344641/SMC/Nr-MASR/data/${split}_oracle_v1/
decode_log=$ckpt_path/decode_${split}_beam4

# -m debugpy --listen 5678 --wait-for-client
srun python $code_dir/inference_mala_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vicuna-7b-v1.5" \
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
        ++dataset_config.use_ocr=true \
        ++dataset_config.dev_scp_file_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=mala_asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \