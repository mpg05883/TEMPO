#!/bin/bash
#SBATCH --job-name=72m2m_np          # Job name
#SBATCH --output=output.6domain_96m2m_no_pool_%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=20G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs
# use the following to remove \r: sed -i 's/\r//g' monash_prob_demo.sh 
export CUDA_VISIBLE_DEVICES=4

seq_len=336
model=TEMPO #TEMPO #PatchTST #_multi
electri_multiplier=3 # 3 times more data than the other small samples.
traffic_multiplier=3


for percent in 100 
do
for pred_len in  96 
do
for tmax in 20
do
for lr in 0.001 
do
for gpt_layer in 6 
do
for equal in 1
do
for prompt in 1 
do
if [ ! -d "logs/$model" ]; then
    mkdir -p logs/$model
    echo "Directory logs/$model has been created."
else
    echo "Directory logs/$model already exists."
fi

dir1="logs/$model/Prob_ReVIN_${prompt}_${prompt}_equal_${equal}/"
if [ ! -d "$dir1" ]; then
    mkdir -p "$dir1"
    echo "Directory $dir1 has been created."
else
    echo "Directory $dir1 already exists."  
fi

dir2="${dir1}Monash_${model}_${gpt_layer}"
if [ ! -d "$dir2" ]; then
    mkdir -p "$dir2"
    echo "Directory $dir2 has been created."
else
    echo "Directory $dir2 already exists."
fi

log_path="${dir2}/test_${seq_len}_${pred_len}_lr${lr}.log"
echo -e "$log_path"

# python train_TEMPO_prob.py \
# MASTER_PORT=29503 torchrun --nproc_per_node=2 train_TEMPO_prob_parallel.py \
python train_TEMPO_prob.py \
    --datasets ETTh2 \
    --eval_data ETTm1 \
    --target_data ETTh2 \
    --config_path ./configs/multiple_datasets.yml \
    --stl_weight 0.001 \
    --equal $equal \
    --checkpoint ./checkpoints/Monash'_'$prompt/ \
    --model_id Demo_Monash_TEMPO_Prob'_'$gpt_layer'_'prompt_learn'_'$seq_len'_'$pred_len'_'$percent \
    --electri_multiplier $electri_multiplier \
    --traffic_multiplier $traffic_multiplier \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --prompt $prompt\
    --batch_size 256 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 1 \
    --patch_size 16 \
    --stride 8 \
    --gpt_layer $gpt_layer \
    --itr 3 \
    --model $model \
    --tmax $tmax \
    --cos 1 \
    --is_gpt 1 \
    --loss_func prob
    #>> logs/$model/Prob_ReVIN_$prompt'_'prompt'_'equal'_'$equal/Monash_$model'_'$gpt_layer/test'_'$seq_len'_'$pred_len'_lr'$lr.log


done
done
done
done
done
done
done
