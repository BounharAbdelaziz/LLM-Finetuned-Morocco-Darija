export CUDA_VISIBLE_DEVICES="2"

model_name="Llama-7B" # "Llama-7B", "Noon-7B", "aragpt2-large", "gpt2-small-arabic"


# python3 train.py
accelerate launch train.py --model_name $model_name