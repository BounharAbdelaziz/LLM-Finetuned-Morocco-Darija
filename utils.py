import gc
from datasets import Dataset
import os
import glob

    
# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def clean_dataset(dataset, target_lang):
    # Safety check: Remove empty rows where the target_lang column is NaN
    print(f'[INFO] Filter out empty examples...')
    
    # Directly filter the dataset
    dataset = dataset.filter(lambda example: example[target_lang] is not None and example[target_lang] != '')
    
    print(f'[INFO] Filtering ended.')
    return dataset

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def get_save_dir_path(HUB, MODEL_NAME, TRAIN_ON_ALL_DATA, TRAIN_ON_ARABIZI_ONLY, TRAIN_ON_ARABIC_LETTERS_ONLY):
        
    suffix = ""
    if TRAIN_ON_ALL_DATA:
        suffix = suffix + "_all_data"
    else:
        suffix = suffix + "_subset_data"
        
    if TRAIN_ON_ARABIZI_ONLY:
        suffix = suffix + "_arabizi_only"
    elif TRAIN_ON_ARABIC_LETTERS_ONLY:
        suffix = suffix + "_arabic_letters_only"
    else:
        suffix = suffix + "subset_all_writings"
    
    save_model_path = HUB + "/" + MODEL_NAME + "/" + suffix

    return save_model_path
    
# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def find_checkpoint_path(base_path):
    
    """ Search for the checkpoint file in the directory """
    # Use glob to search for the checkpoint file
    checkpoint_pattern = os.path.join(base_path, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if checkpoints:
        # Assuming there's only one checkpoint file
        return checkpoints[0]
    else:
        raise FileNotFoundError("No checkpoint file found in the directory.")
    
# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def preprocess_function(examples, tokenizer, max_seq_length=1024, text_field="text"):
    tokenized = tokenizer(
        examples[text_field],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors=None, # returns a list #"pt"
    )
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized

# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #