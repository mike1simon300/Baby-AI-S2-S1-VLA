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

    # Format examples: concatenate instruction and response.
    def format_example(example):
        return {
            "text": f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"
        }

    train_dataset = train_dataset.map(format_example)
    eval_dataset = eval_dataset.map(format_example)

    # Remove extra columns, ensuring each example has a "text" field.
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col != "text"])
    eval_dataset = eval_dataset.remove_columns([col for col in eval_dataset.column_names if col != "text"])

    # Debug: print first 3 formatted training examples
    print("Printing first 3 formatted training examples for inspection:")
    for i in range(3):
        print(f"Example {i}:")
        print(train_dataset[i]["text"])
        print("-----")

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
    # Ensure correct types for training config
    training_config["learning_rate"] = float(training_config["learning_rate"])
    training_config["weight_decay"] = float(training_config["weight_decay"])
    training_config["warmup_steps"] = int(training_config["warmup_steps"])
    training_config["num_train_epochs"] = int(training_config["num_train_epochs"])
    training_config["eval_steps"] = int(training_config["eval_steps"])
    training_config["gradient_accumulation_steps"] = int(training_config["gradient_accumulation_steps"])
    training_config["per_device_train_batch_size"] = int(training_config["per_device_train_batch_size"])
    training_config["per_device_eval_batch_size"] = int(training_config["per_device_eval_batch_size"])

    training_args = SFTConfig(**training_config)

    # Custom data collator: if a feature has no "text" key, decode its input_ids.
    # Then, split the raw text at the delimiter "### Response:" and mask out the prompt tokens.
    def custom_data_collator(features):
        texts = []
        for i, f in enumerate(features):
            if "text" in f:
                texts.append(f["text"])
            else:
                # Decode the already-tokenized inputs to recover the raw text.
                decoded = tokenizer.decode(f["input_ids"], skip_special_tokens=False)
                texts.append(decoded)
        batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        labels = batch["input_ids"].clone()

        delimiter = "### Response:"
        for i, text in enumerate(texts):
            if delimiter in text:
                prompt_part = text.split(delimiter)[0]
                prompt_ids = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
                prompt_length = len(prompt_ids)
                labels[i, :prompt_length] = -100
            else:
                print(f"Warning: Delimiter '{delimiter}' not found in example {i}.")
        batch["labels"] = labels
        return batch

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()
