import torch
import random
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned LoRA adapter")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the inference dataset (Arrow format)")
    parser.add_argument("--index", type=int, default=None, help="Optional index of the example to run. If not provided, a random one will be selected.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p value (default: 0.9)")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("🚨 Running on CPU 🚨")

    print(f"📂 Loading inference dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)

    if len(dataset) == 0:
        print("❌ The dataset is empty!")
        return

    # Select example
    if args.index is not None:
        if args.index < 0 or args.index >= len(dataset):
            print(f"❌ Invalid index: {args.index}. Dataset size: {len(dataset)}")
            return
        idx = args.index
    else:
        idx = random.randint(0, len(dataset) - 1)

    example = dataset[idx]
    prompt = f"### Instruction:\n{example['input']}\n\n### Response:"

    print(f"\n✅ Using example index: {idx}")
    print("📜 Selected prompt:\n")
    print(prompt)

    # Load tokenizer
    print(f"\n🚀 Loading tokenizer from base model")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    # Load base model explicitly onto CPU
    print(f"🔧 Loading base model and LoRA adapter from {args.model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        device_map="cpu",
        torch_dtype=torch.float32,  # CPU inference should use float32
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model = model.to(device)

    model.eval()

    def generate_text(prompt, max_length=2048):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_response = decoded_output[len(prompt):].strip()

        return generated_response

    print("\n🔮 Generating output...\n")
    result = generate_text(prompt)
    print("📝 Generated Response:\n")
    print(result)


if __name__ == "__main__":
    main()
