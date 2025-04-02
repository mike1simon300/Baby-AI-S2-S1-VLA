import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Optionally, clear CUDA cache to free up memory
torch.cuda.empty_cache()

# Set dataset path (update as needed)
dataset_path = "./datasets/robot_LLM_grid_dataset_15k_merged/mixed_dataset"
dataset = load_from_disk(dataset_path)

# Preprocess the dataset to ensure each example has a unified "text" field
if "text" not in dataset.column_names:
    def format_example(example):
        # Adjust the prompt format if needed
        return {"text": f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"}
    dataset = dataset.map(format_example, batched=False)

# Load tokenizer and model
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure BitsAndBytes for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# Load the model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Prepare the model for k-bit training (required for LoRA with quantized models)
model = prepare_model_for_kbit_training(model)

# Set up LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the model with LoRA adapters
model = get_peft_model(model, lora_config)

# Define training configuration using SFTConfig from TRL
training_args = SFTConfig(
    output_dir="./gemma-2b-it-finetuned",
    per_device_train_batch_size=1,         # Adjust as needed
    gradient_accumulation_steps=16,        # Effective batch size = 1 x 16
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,
    dataloader_num_workers=4,
    max_length=1024,  # You can reduce this if memory issues persist
    packing=False
)

# Initialize the trainer (note: do not pass tokenizer here as it's not accepted)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Start training
trainer.train()
