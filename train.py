import argparse
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
import logging
from utils import get_save_dir_path, clean_dataset, find_checkpoint_path, preprocess_function

# Ignore warnings
# logging.set_verbosity(logging.CRITICAL)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Trainer of an LLM that speaks Moroccan Darija.')
    parser.add_argument('--model_name', required=True, type=str, help='Model name: "smollm2-1.7B", "Gemma-2-2B", "Llama-7B", "Noon-7B", "aragpt2-large", "gpt2-small-arabic"')
    parser.add_argument('--dataset_name', required=True, type=str, default ='fineweb_filtered_sawalni_ai', help='')
    parser.add_argument('--num_train_epochs', required=True, type=int, default =10, help='')
    parser.add_argument('--batch_size', required=True, type=int, default =8, help='')
    parser.add_argument('--learning_rate', required=True, type=float, default =5e-3, help='')
    parser.add_argument('--logging_steps', required=True, type=int, default =1000, help='')
    parser.add_argument('--save_steps', required=True, type=int, default =0, help='')
    parser.add_argument('--use_lora', action="store_true", help='Use LORA finetuning flag')
    args = parser.parse_args()

    # The Hugging Face API to push model to the hub
    HF_API = HfApi()

    # Training hyperparameters and arguments
    MODEL_NAME = args.model_name
    DATASET_NAME = args.dataset_name
    num_train_epochs = args.num_train_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    save_steps = args.save_steps
    logging_steps = args.logging_steps
    USE_LORA = args.use_lora
    max_seq_length = 1024
    FROM_SCATCH = True

    if USE_LORA:
        print(f'[INFO] LoRA finetuning')
    else:
        print(f'[INFO] full finetuning')

    # GPU device map
    device_map = {'': torch.cuda.current_device()}
    print("=" * 80)
    print('Current device:', torch.cuda.current_device())

    #### These params must go to a yaml config file
    # Set data parameters
    TRAIN_ON_ALL_DATA = True

    # Column name of the target language
    target_lang = "text"

    # The dataset to use
    DATA_PATHS = {
        "my_mix": "BounharAbdelaziz/AL-Atlas-Moroccan-Darija-Pretraining-Dataset",
        "fineweb_filtered_sawalni_ai": "sawalni-ai/fw-darija"
    }
    
    DATA_PATH = DATA_PATHS[DATASET_NAME]

    # Model paths
    MODEL_PATHS = {
                    "Gemma-2-2B":           "google/gemma-2-2b",
                    "Llama-7B" :            "sambanovasystems/SambaLingo-Arabic-Base",
                    "Noon-7B":              "Naseej/noon-7b",
                    "aragpt2-large" :       "aubmindlab/aragpt2-large",
                    "gpt2-small-arabic" :   "akhooli/gpt2-small-arabic",
                    "smollm2-1.7B" :        "HuggingFaceTB/SmolLM2-1.7B",
                    "smollm2-360M" :        "HuggingFaceTB/SmolLM2-360M",
                    "smollm2-135M" :        "HuggingFaceTB/SmolLM2-135M",
                    "Qwen2.5-0.5B":         "Qwen/Qwen2.5-0.5B",
                    "Qwen2.5-1.5B":         "Qwen/Qwen2.5-1.5B",
                    "hf-ar-107000":         "nouamanetazi/hf-ar-107000",
                    "mine-0.5B":            "Qwen/Qwen2.5-0.5B",

                }

    TOKENIZER_PATH = "asafaya/bert-base-arabic"

    # Set path of base model, where to save the fine-tuned model, and dataset to use for fine-tuning
    MODEL_PATH = MODEL_PATHS[MODEL_NAME]

    # Finetuned model name
    if MODEL_NAME == "smollm2-1.7B":
        MY_MODEL_NAME = "Al-Atlas-LLM-Large"
        HUB = "BounharAbdelaziz/Al-Atlas-LLM-Large"
        SAVE_MODEL_PATH = "BounharAbdelaziz/Al-Atlas-LLM-Large"

    elif MODEL_NAME == "smollm2-360M":
        MY_MODEL_NAME = "Al-Atlas-LLM-Medium"
        HUB = "BounharAbdelaziz/Al-Atlas-LLM-Medium"
        SAVE_MODEL_PATH = "BounharAbdelaziz/Al-Atlas-LLM-Medium"
    
    elif MODEL_NAME == "smollm2-135M":
        MY_MODEL_NAME = "Al-Atlas-LLM-Small-sawalni-filtered-data-v2"
        HUB = "BounharAbdelaziz/Al-Atlas-LLM-Small-sawalni-filtered-data-v2"
        SAVE_MODEL_PATH = "BounharAbdelaziz/Al-Atlas-LLM-Small-sawalni-filtered-data-v2"
        
    elif MODEL_NAME == "Qwen2.5-0.5B":
        MY_MODEL_NAME = "Al-Atlas-LLM-Medium-Q-0.5B"
        HUB = "BounharAbdelaziz/Al-Atlas-LLM-Medium-Q-0.5B"
        SAVE_MODEL_PATH = "BounharAbdelaziz/Al-Atlas-LLM-Medium-Q-0.5B"
    
    elif MODEL_NAME == "Qwen2.5-1.5B":      
        MY_MODEL_NAME = "Al-Atlas-LLM-Medium-Q-1.5B"
        HUB = "BounharAbdelaziz/Al-Atlas-LLM-Medium-Q-1.5B"
        SAVE_MODEL_PATH = "BounharAbdelaziz/Al-Atlas-LLM-Medium-Q-1.5B"

    elif MODEL_NAME == "Gemma-2-2B":
        MY_MODEL_NAME = "Al-Atlas-LLM-G2-2B"
        HUB = "BounharAbdelaziz/Al-Atlas-LLM-G2-2B"
        SAVE_MODEL_PATH = "BounharAbdelaziz/Al-Atlas-LLM-G2-2B"
        
    elif MODEL_NAME == "gpt2-small-arabic":
        MY_MODEL_NAME = "Al-Atlas-LLM-Tiny"
        HUB = "BounharAbdelaziz/Al-Atlas-LLM-Tiny"
        SAVE_MODEL_PATH = "BounharAbdelaziz/Al-Atlas-LLM-Tiny"
        
    elif MODEL_NAME == "hf-ar-107000":
        MY_MODEL_NAME = "Al-Atlas-LLM-Ultra-v2"
        HUB = "BounharAbdelaziz/Al-Atlas-LLM-Ultra-v2"
        SAVE_MODEL_PATH = "BounharAbdelaziz/Al-Atlas-LLM-Ultra-v2"   

    elif MODEL_NAME == "mine-0.5B":
        MY_MODEL_NAME = "Al-Atlas-LLM-0.5B"
        HUB = "BounharAbdelaziz/Al-Atlas-LLM-0.5B"
        SAVE_MODEL_PATH = "BounharAbdelaziz/Al-Atlas-LLM-0.5B"          

    print("=" * 80)
    print(f'[INFO] Will fine-tune the model {MODEL_PATH} for {num_train_epochs} epochs with a batch size of {batch_size} on the dataset {DATA_PATH} and save the model to {SAVE_MODEL_PATH} ...')
    print("=" * 80)

    # Load dataset
    if TRAIN_ON_ALL_DATA:
        print('[INFO] Loading the entire dataset')
        print("=" * 80)
        dataset = load_dataset(DATA_PATH, split="train[:1%]") # [1%:2%]
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
    # LoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 256

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.15

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = True
    
    # Batch size per GPU for training
    per_device_train_batch_size = batch_size

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = batch_size

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.9

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.05

    # Optimizer to use
    optim = "adamw_torch"

    # Learning rate schedule
    lr_scheduler_type = "linear"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    group_by_length = True

    # Put the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ################################################################################
    # SFT parameters
    ################################################################################

    if FROM_SCATCH:
        print(f'[INFO] Training from scratch, loading config...')
        config = AutoConfig.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_config(config,
                                                torch_dtype=torch.float16  # Use FP16 training
        )  
        # Move the model to the specified device
        model.to(device)
    else:
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map=device_map,
            torch_dtype=torch.float16  # Use FP16 training
        )

    # Load LoRA configuration
    if USE_LORA:
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"
    
    # Apply preprocessing
    print(f'[INFO] Tokenizing dataset...')
    tokenized_dataset = dataset.map(
            lambda x: preprocess_function(x, tokenizer, max_seq_length=max_seq_length, text_field=target_lang),
            batched=True,

        )        
    print(f'[INFO] Dataset tokenization ended.')

    # Use a DataCollator with padding
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt") # Set to True for masked language modeling

    # Set training parameters
    training_args = TrainingArguments(
        gradient_checkpointing=gradient_checkpointing,
        output_dir=SAVE_MODEL_PATH,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard",
        push_to_hub=True,  # Enable push to Hub
        hub_model_id=MY_MODEL_NAME,  # Name of your model on the Hub
        save_total_limit=1,  # Only keep the last checkpoint
        save_strategy="steps",  # Save model periodically
    )

    # Set supervised fine-tuning parameters
    if USE_LORA:
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field=target_lang,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
        )

    # Train model
    print("[INFO] Started training...")
    trainer.train()

    print(f"[INFO] Training ended successfully, saving model in {SAVE_MODEL_PATH}...")

    # Reload base model to merge with LoRA weights
    if USE_LORA:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map='auto',
        )

        # merge with LoRA weights
        LAST_CHECKPOINT = find_checkpoint_path(SAVE_MODEL_PATH)
        model = PeftModel.from_pretrained(base_model, SAVE_MODEL_PATH)
        model = model.merge_and_unload()

    # Save model and tokenizer
    tokenizer.save_pretrained(SAVE_MODEL_PATH)
    model.save_pretrained(SAVE_MODEL_PATH)

    # Push model to Hugging Face Hub
    trainer.push_to_hub()

    print(f"[INFO] Model successfully saved and pushed to {HUB}!")
    print("=" * 80)
