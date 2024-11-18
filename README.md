# LLM-Finetuned-Morocco-Darija

This repository provides the code used to finetune a Large Language Model (LLM) for Moroccan Darija. Specifically, we finetuned a LLaMA-2 7B [arabic version](https://huggingface.co/sambanovasystems/SambaLingo-Arabic-Base) using efficient finetuning techniques, here for instance **QLoRA** for 4-bit quantization.
The code also supports other base models, including LLaMA, Noon, and Arabic-specific GPT models, and any other model from [Hugging Face](https://huggingface.co/).
The finetuning was performed using using a **A100-40GB** GPU.

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/BounharAbdelaziz/LLM-Finetuned-Morocco-Darija.git
   cd LLM-Finetuned-Morocco-Darija
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Hugging Face authentication:
   ```bash
   huggingface-cli login
   ```

4. Adjust the `utils.py` file to define the `get_save_dir_path` and `clean_dataset` functions as per your dataset structure.

5. Redefine the hyperparameters based on your computes configuration.

## Usage

### Training the Model

Run the training script with the following command:
```bash
python train.py --model_name "Llama-7B"
```

### Arguments:
- `--model_name`: Select the model to fine-tune (e.g., "Llama-7B").


## Project Structure

```
LLM-Finetuned-Morocco-Darija/
├── train.py           # Main training script
├── test.py            # Testing script
├── utils.py           # Utility functions for data cleaning and path setup
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
```

## Customization

### Supported Models
Modify the `MODEL_PATHS` dictionary in `train.py` to add or replace model configurations.

### Dataset Cleaning
Update the `clean_dataset()` function in `utils.py` to adjust preprocessing logic, such as removing unwanted characters or handling Arabizi formats.


## Acknowledgments, Feedback & Limitations

This project is part of ongoing efforts to advance Moroccan Darija NLP, leveraging state-of-the-art machine learning techniques. Thanks to the open-source AI community and Hugging Face for providing valuable resources.