import argparse
import yaml
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Load config from YAML file
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Clear cache
    torch.cuda.empty_cache()

    # Load and split dataset
    dataset = load_from_disk(config["dataset_path"])
    dataset = dataset.train_test_split(test_size=config["test_size"])
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    def format_example(example):
        return {
            "text": f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"
        }

    train_dataset = train_dataset.map(format_example)
    eval_dataset = eval_dataset.map(format_example)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=config["bnb_config"]["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, config["bnb_config"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=config["bnb_config"]["bnb_4bit_quant_type"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        lora_dropout=config["lora_config"]["lora_dropout"],
        bias=config["lora_config"]["bias"],
        task_type=config["lora_config"]["task_type"],
    )

    model = get_peft_model(model, lora_cfg)

    training_config = config["training"]

    # Ensure correct types
    training_config["learning_rate"] = float(training_config["learning_rate"])
    training_config["weight_decay"] = float(training_config["weight_decay"])
    training_config["warmup_steps"] = int(training_config["warmup_steps"])
    training_config["num_train_epochs"] = int(training_config["num_train_epochs"])
    training_config["eval_steps"] = int(training_config["eval_steps"])
    training_config["gradient_accumulation_steps"] = int(training_config["gradient_accumulation_steps"])
    training_config["per_device_train_batch_size"] = int(training_config["per_device_train_batch_size"])
    training_config["per_device_eval_batch_size"] = int(training_config["per_device_eval_batch_size"])

    training_args = SFTConfig(**config["training"])

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()
