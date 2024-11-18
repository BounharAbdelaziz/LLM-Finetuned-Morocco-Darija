import os
os.environ["TORCH_USE_CUDA_DSA"] = "1" # Enable CUDA DSA
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch


if __name__ == "__main__":

    # GPU device map
    device_map={'':torch.cuda.current_device()}
    print('Current device:', torch.cuda.current_device())
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("BounharAbdelaziz/Al-Atlas-LLM")
    model = AutoModelForSeq2SeqLM.from_pretrained("BounharAbdelaziz/Al-Atlas-LLM")

    # Run text generation pipeline with our next model
    prompt = "Wach t9der"
    print(f"[INFO] An example of output, the prompt is: <s> {prompt}...")
    print("=" * 80)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=256)
    result = pipe(f"<s> {prompt}")
    print(result[0]['generated_text'])
    
    
    # Run text generation pipeline with our next model
    prompt = "ﺃﻧﺎ كنستغرب ﺻﺮﺍﺣﺔ علاش مزال عايش"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=256)
    result = pipe(f"{prompt}")
    print(result[0]['generated_text'])
    
    prompt = "Wach t9der"
    print(f"[INFO] An example of output, the prompt is {prompt}...")
    print("=" * 80)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=256)
    result = pipe(f"{prompt}")
    print(result[0]['generated_text'])
    
    


    