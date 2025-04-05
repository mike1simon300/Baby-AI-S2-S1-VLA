import torch
import random
import argparse
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the inference dataset (Arrow format)")
    parser.add_argument("--index", type=int, default=None, help="Optional index of the example to run. If not provided, a random one will be selected.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p value (default: 0.9)")
    args = parser.parse_args()

    print(f"Loading inference dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)

    if len(dataset) == 0:
        print("The dataset is empty!")
        return

    if args.index is not None:
        if args.index < 0 or args.index >= len(dataset):
            print(f"Invalid index: {args.index}. Dataset size: {len(dataset)}")
            return
        idx = args.index
    else:
        idx = random.randint(0, len(dataset) - 1)

    example = dataset[idx]
    prompt = f"### Instruction:\n{example['input']}\n\n### Response:"

    print(f"\nUsing example index: {idx}")
    print("Selected prompt:\n")
    print(prompt)

    print("\nLoading model and tokenizer from", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    def generate_text(prompt, max_length=2048):
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(output[0], skip_special_tokens=True)

    print("\nGenerating output...\n")
    result = generate_text(prompt)
    print("Generated Output:\n")
    print(result)

if __name__ == "__main__":
    main()