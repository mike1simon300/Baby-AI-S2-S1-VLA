import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Clear CUDA cache
torch.cuda.empty_cache()

# Load and split dataset
dataset_path = "./datasets/robot_LLM_grid_dataset_15k_merged/mixed_dataset"
dataset = load_from_disk(dataset_path)

# Create 90/10 train/val split
dataset = dataset.train_test_split(test_size=0.03)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Format each example as instruction-response pair
def format_example(example):
    return {
        "text": f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"
    }

train_dataset = train_dataset.map(format_example)
eval_dataset = eval_dataset.map(format_example)

# Load model and tokenizer
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training configuration
training_args = SFTConfig(
    output_dir="./gemma-2b-it-finetuned_2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",                 # Logs go here
    logging_steps=1,                     # Log every step
    report_to="tensorboard",             # Enable TensorBoard logging
    evaluation_strategy="steps",
    per_device_eval_batch_size=1,
    eval_steps=50,                       # Validate every 50 steps
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,
    max_length=1024,
    packing=False
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start training
trainer.train()
