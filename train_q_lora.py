import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from utils import get_save_dir_path, clean_dataset

# Ignore warnings
# logging.set_verbosity(logging.CRITICAL)

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Trainer of an LLM that speaks Moroccan Darija.')
    parser.add_argument('--model_name', required=True, type=str, help='Model name: "Llama-7B", "Noon-7B", "aragpt2-large", "gpt2-small-arabic"')
    args = parser.parse_args()

    # The Hugging Face API to push model to the hub
    HF_API = HfApi()

    # Training hyperparameters and arguments
    MODEL_NAME = args.model_name

    # GPU device map
    device_map = {'': torch.cuda.current_device()}
    print("=" * 80)
    print('Current device:', torch.cuda.current_device())

    # Batch size for training
    batch_size = 32
    num_train_epochs = 1
    learning_rate = 2e-4
    save_steps = 0
    logging_steps = 100

    # Set data parameters
    TRAIN_ON_ALL_DATA = True
    TRAIN_ON_ARABIZI_ONLY = False
    TRAIN_ON_ARABIC_LETTERS_ONLY = True

    # Column name of the target language
    target_lang = "text"

    # The dataset to use
    DATA_PATH = "BounharAbdelaziz/AL-Atlas-Moroccan-Darija-Pretraining-Dataset"

    # Model paths
    MODEL_PATHS = {
                    "Gemma-2-2B":           "google/gemma-2-2b",
                    "Llama-7B" :            "sambanovasystems/SambaLingo-Arabic-Base",
                    "Noon-7B":              "Naseej/noon-7b",
                    "aragpt2-large" :       "aubmindlab/aragpt2-large",
                    "gpt2-small-arabic" :   "akhooli/gpt2-small-arabic",
                }

    # Finetuned model name
    HUB = "BounharAbdelaziz/Al-Atlas-LLM"
    
    # Set path of base model, where to save the fine-tuned model, and dataset to use for fine-tuning
    MODEL_PATH = MODEL_PATHS[MODEL_NAME]
    
    # Where to save the weights
    SAVE_MODEL_PATH = get_save_dir_path(HUB, MODEL_NAME, TRAIN_ON_ALL_DATA, TRAIN_ON_ARABIZI_ONLY, TRAIN_ON_ARABIC_LETTERS_ONLY)
    
    # Output directory where the model predictions and checkpoints will be stored
    output_dir = os.path.join(SAVE_MODEL_PATH, "results")

    print("=" * 80)
    print(f'[INFO] Will fine-tune the model {MODEL_PATH} for {num_train_epochs} epochs with a batch size of {batch_size} on the dataset {DATA_PATH} and save the model to {SAVE_MODEL_PATH} ...')
    print("=" * 80)

    # Load dataset
    if TRAIN_ON_ALL_DATA:
        print('[INFO] Loading the entire dataset')
        print("=" * 80)
        dataset = load_dataset(DATA_PATH, split="train")
    else:
        print('[INFO] Loading only a subset of the dataset')
        print("=" * 80)
        dataset = load_dataset(DATA_PATH, split='train[50%:60%]')
        
    print('[INFO] Cleaning the dataset...')
    print("=" * 80)
    dataset = clean_dataset(dataset, target_lang)
    print(f'[INFO] Dataset cleaning ended successfully! Training on {len(dataset)} examples...')
    print("=" * 80)

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = True  # Use FP16 training for faster processing

    # Batch size per GPU for training
    per_device_train_batch_size = batch_size

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = batch_size

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule
    lr_scheduler_type = "cosine"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
            bf16 = True

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map=device_map
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_args = TrainingArguments(
        gradient_checkpointing=gradient_checkpointing,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=target_lang,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=packing,
    )

    # Train model
    with torch.profiler.profile() as prof:
        print("[INFO] Started training...")
        print("=" * 80)
        trainer.train()

        prof.export_chrome_trace("trace.json")

    print(f"[INFO] Training ended successfully, saving model in {SAVE_MODEL_PATH} and pushing to hub {HUB}...")
    print("=" * 80)
    
    # Reload model in FP16 and merge it with LoRA weights
    print(f"[INFO] Reloading model in FP16 and merge it with LoRA weights...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    model = PeftModel.from_pretrained(base_model, SAVE_MODEL_PATH)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Save the merged model locally
    print(f"[INFO] Saving and pushing the merged model to {HUB}...")
    model.save_pretrained(SAVE_MODEL_PATH)
    tokenizer.save_pretrained(SAVE_MODEL_PATH)
    
    # Push model to Hugging Face Hub
    HF_API.upload_folder(
        repo_id=HUB,
        folder_path=SAVE_MODEL_PATH,
        repo_type="model",
        create_pr=False  # Set to True if you want to open a PR instead of pushing directly
    )
    
    print(f"[INFO] Model successfully saved and pushed to {HUB}!")
    print("=" * 80)