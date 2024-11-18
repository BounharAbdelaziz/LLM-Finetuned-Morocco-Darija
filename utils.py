import gc
from datasets import Dataset
    
# ------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------ #

def clean_dataset(dataset, target_lang):

    # transform to pandas dataframe
    df = dataset.to_pandas()
    
    # Safety check: Remove empty rows
    print(f'[INFO] Filter out empty examples...')
    df = df.dropna(subset=[target_lang])
    dataset = Dataset.from_pandas(df)
    print(f'[INFO] Filtering ended.')
    
    dataset = Dataset.from_pandas(df)
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