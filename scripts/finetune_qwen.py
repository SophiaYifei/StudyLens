# scripts/finetune.py
# QLoRA fine-tuning for Qwen2.5-7B-Instruct
# Run in Colab LLM environment with A100 GPU
#
# Usage (in Colab after mounting Drive and cd to project root):
#   !pip install -r requirements-llm.txt
#   !pip install peft trl datasets
#
#   # Step 1: Prepare training data
#   !python scripts/finetune.py --mode prepare
#
#   # Step 2: Train
#   !python scripts/finetune.py --mode train --epochs 3
#
#   # Step 3: Run inference with fine-tuned model
#   !python scripts/finetune.py --mode inference

import json
import torch
import os
from pathlib import Path

# --- Paths ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
REFERENCE_DIR = DATA_DIR / "reference"
TRAINING_JSON = OUTPUTS_DIR / "finetune" / "ft_training_data.json"
ADAPTER_DIR = _PROJECT_ROOT / "models" / "qwen7b-lora"


# ============================================================
# Step 1: Prepare training data
# ============================================================

def prepare_training_data():
    """
    Build training JSON from transcript + reference summary pairs.

    Expected files:
        data/processed/{topic}_ori.txt         (input transcripts)
        data/reference/{topic}_ref.txt   (ChatGPT-generated reference summaries)

    The 2 test topics should NOT have reference files —
    only create references for the 8 training topics.
    """

    # Same prompt used in QwenSummarizer — keeps training consistent with inference
    instruction = (
        "Summarize the following university lecture material into a comprehensive "
        "yet concise summary. The input combines slide text, lecture transcript, and student "
        "notes, so it contains filler words, informal language, typos, and "
        "off-topic conversations — ignore all of these.\n\n"
        "Your summary should:\n"
        "- Cover every major topic and subtopic discussed\n"
        "- Include key definitions, formulas, and technical details\n"
        "- Preserve the logical flow of the lecture\n"
        "- Use clear academic language"
    )

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_JSON.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    for ori_file in sorted(PROCESSED_DIR.glob("*_ori.txt")):
        topic_key = ori_file.stem.replace("_ori", "")
        # Skip test topics — these are held out for final evaluation
        TEST_TOPICS = {"dl_s5", "ml_s5"}
        if topic_key in TEST_TOPICS:
            print(f"  SKIP (test set): {topic_key}")
            continue
        ref_file = REFERENCE_DIR / f"{topic_key}_ref.txt"

        if not ref_file.exists():
            print(f"  SKIP (no reference): {topic_key}")
            continue

        transcript = ori_file.read_text(encoding="utf-8")
        summary = ref_file.read_text(encoding="utf-8")

        samples.append({
            "instruction": instruction,
            "input": transcript,
            "output": summary,
        })
        print(f"  Added: {topic_key} "
              f"({len(transcript.split())} words -> {len(summary.split())} words)")

    # Save
    TRAINING_JSON.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved {len(samples)} training samples to {TRAINING_JSON}")


# ============================================================
# Step 2: Fine-tune
# ============================================================

def finetune(epochs=3, val_ratio=0.1):
    """
    QLoRA fine-tuning of Qwen2.5-7B-Instruct.

    What happens:
    1. Load base model with 4-bit quantization (same as QwenSummarizer)
    2. Freeze all 7B parameters
    3. Insert small LoRA adapter matrices (~0.1% of total params)
    4. Train only the adapters on the data
    5. Monitor eval_loss on validation split to pick best checkpoint
    6. Save adapter weights (~50MB)
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig    
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import Dataset

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # --- Load base model (same config as QwenSummarizer.__init__) ---
    print("Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # needed for batched training

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # --- Freeze base model, enable gradient checkpointing ---
    model = prepare_model_for_kbit_training(model)

    # --- Configure LoRA adapter ---
    lora_config = LoraConfig(
        r=16,                               # rank of adapter matrices
        lora_alpha=32,                      # scaling factor (alpha/r = 2x)
        target_modules=[                    # add adapters to attention layers
            "q_proj", "k_proj",
            "v_proj", "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # --- Load and format training data ---
    print("Loading training data...")
    with open(TRAINING_JSON, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    def format_sample(sample):
        """Convert to Qwen chat format for training."""
        messages = [
            {"role": "system", "content": (
                "You are an expert academic summarizer specializing in computer "
                "science and deep learning courses. You produce clear, "
                "well-organized summaries from university lecture materials."
            )},
            {"role": "user", "content": (
                sample["instruction"] + "\n\nINPUT TEXT:\n" + sample["input"]
            )},
            {"role": "assistant", "content": sample["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    # --- Train/Val split ---
    split = dataset.train_test_split(test_size=val_ratio, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    print(f"Training samples: {len(train_dataset)}, "
          f"Validation samples: {len(val_dataset)}")

    # --- Training config ---
    training_args = SFTConfig(
        output_dir="./qwen-finetune-checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=2,
        logging_steps=1,
        bf16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        # Validation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # SFT-specific (moved from SFTTrainer args)
        max_length=4096,
    )
    
    # --- Train ---
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()

    # --- Print training history ---
    print("\n--- Training History ---")
    for entry in trainer.state.log_history:
        if "eval_loss" in entry:
            print(f"  Epoch {entry.get('epoch', '?'):.1f}: "
                  f"eval_loss = {entry['eval_loss']:.4f}")
        elif "loss" in entry:
            print(f"  Step {entry.get('step', '?')}: "
                  f"train_loss = {entry['loss']:.4f}")

    print("\nTraining complete!")

    # --- Save best adapter ---
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving best adapter to {ADAPTER_DIR}...")
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print("Adapter saved!")


# ============================================================
# Step 3: Inference with fine-tuned model
# ============================================================

def run_finetuned_inference():
    """
    Load base model + LoRA adapter, run inference on all topics,
    save outputs to data/outputs/neural_network/qwen7b-ft/final/
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    print("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Load LoRA adapter on top of base model
    print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    max_input_tokens = 65536
    output_ratio = 0.06

    # Output directory
    output_path = OUTPUTS_DIR / "finetune" / "qwen7b-ft" / "final"
    output_path.mkdir(parents=True, exist_ok=True)

    # Process all topics
    ori_files = sorted(PROCESSED_DIR.glob("*_ori.txt"))
    print(f"Found {len(ori_files)} files to process.\n")

    for file in ori_files:
        text = file.read_text(encoding="utf-8")
        total_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        print(f"Processing: {file.name} ({total_tokens} tokens)")

        # Truncate if needed
        if total_tokens > max_input_tokens:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            text = tokenizer.decode(tokens[:max_input_tokens])

        # Build prompt (same as QwenSummarizer.summarize)
        messages = [
            {"role": "system", "content": (
                "You are an expert academic summarizer specializing in computer "
                "science and deep learning courses. You produce clear, "
                "well-organized summaries from university lecture materials."
            )},
            {"role": "user", "content": (
                "Summarize the following university lecture material into a comprehensive "
                "yet concise summary. The input combines slide text, lecture transcript, and student "
                "notes, so it contains filler words, informal language, typos, and "
                "off-topic conversations — ignore all of these.\n\n"
                "Your summary should:\n"
                "- Cover every major topic and subtopic discussed\n"
                "- Include key definitions, formulas, and technical details\n"
                "- Preserve the logical flow of the lecture\n"
                "- Use clear academic language\n\n"
                f"INPUT TEXT:\n{text}"
            )},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Dynamic output length (same as QwenSummarizer)
        target_output = int(total_tokens * output_ratio)
        max_new = max(300, target_output)
        min_new = max(150, max_new // 4)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Save
        output_filename = file.name.replace("_ori.txt", "_sum_qwen7b-ft_final.txt")
        output_file = output_path / output_filename
        output_file.write_text(summary, encoding="utf-8")
        print(f"  Saved: {output_file.name} ({len(summary.split())} words)\n")

    print("Done! Fine-tuned outputs saved.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StudyLens QLoRA Fine-tuning")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["prepare", "train", "inference"],
                        help="prepare: build training JSON; "
                             "train: run QLoRA fine-tuning; "
                             "inference: run fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_training_data()
    elif args.mode == "train":
        finetune(epochs=args.epochs)
    elif args.mode == "inference":
        run_finetuned_inference()