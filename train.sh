export CUDA_VISIBLE_DEVICES="1,2"

model_name="Gemma-2-2B" #"mine-0.5B" #"smollm2-135M" #"Qwen2.5-0.5B" #"Gemma-2-2B" # "smollm2-360M" # "smollm2-1.7B" # "Gemma-2-2B", "Llama-7B", "Noon-7B", "aragpt2-large", "gpt2-small-arabic"
dataset_name="my_mix" #"my_mix" #"fineweb_filtered_sawalni_ai"
num_train_epochs=1
batch_size=8
learning_rate=0.0005
save_steps=1000
logging_steps=100

# --use_lora 
# python3 train.py --use_lora --model_name $model_name --dataset_name $dataset_name --num_train_epochs $num_train_epochs \
python3 train.py --model_name $model_name --dataset_name $dataset_name --num_train_epochs $num_train_epochs \
                    --batch_size $batch_size --learning_rate $learning_rate --save_steps $save_steps --logging_steps $logging_steps
# accelerate launch train.py --model_name $model_name