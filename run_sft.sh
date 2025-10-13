
export RAY_TMPDIR="/tmp/ray_tmp"

module load GCC/13.3.0
export CC=$(which gcc)
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDNN_HOME=~/libs/cudnn-9.8
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_HOME/include:$CPATH


export WANDB_MODE=offline
export WANDB_PROJECT=sft_Llama-3.2-3B-Instruct_summary


export CUDA_VISIBLE_DEVICES=0,1,2,3



export DATA_DIR=/path/to/verl/train_dataset/sft_ex_sum_full_implicit
export MODEL_DIR=/path/to/verl/HF_models_datasets/models/Llama-3.2-3B-Instruct
export OUTPUT_DIR=/path/to/verl/output_model/sft_Llama-3.2-3B-Instruct_summary_full_1epoch-im
export PROJECT_NAME=Llama-3.2-3B-Instruct-Instruct-OPV2-SFT-summary-im
export EXPERIMENT_NAME=Llama-3.2-3B-Instruct-OPV2-SFT-summary-im

torchrun --nproc_per_node=4 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.prompt_key=flattened_prompt \
    data.response_key=flattened_response \
    data.max_length=1300 \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=${MODEL_DIR} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=1 \
    trainer.logger="['console']" \
    trainer.default_local_dir=${OUTPUT_DIR}
