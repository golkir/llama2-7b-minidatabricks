import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import login

# device
device = "cuda"
# The model to fine-tune
model_name = "meta-llama/Llama-2-7b-hf"

# The instruction dataset to use
dataset_name = "kirillgoltsman/databricks-dolly-llama2-1k"

# Fine-tuned model name
new_model_name = "llama-2-7b-minidatabricks"

# Activate 4-bit precision base model loading
use_4bit = True

# Load model on the GPU 0
device_map = {"": 0}

# Get user input from the console
access_token = input("Enter your HuggingFace access token")

# Convert the input to the desired data type (e.g., int or float)
if access_token == "":
    print("you need to provide HuggingFace access token")
    sys.exit(1)

login(token=access_token)


# Prompt generating

def generate_prompt(model, tokenizer):
    # Run text generation pipeline with a given model
    prompt = "Can you recommend me a movie?"
    pipe = pipeline(task="text-generation", model=model,
                    tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])


# Load dataset
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit precision model loading
    bnb_4bit_quant_type="nf4",  # Quantization type (fp4 or nf4)
    bnb_4bit_compute_dtype=compute_dtype,
    # Activate double (nested) quantization for 4-bit base models
    bnb_4bit_use_double_quant=False,
)


# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training


# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,  # alpha parameter for LoRA scaling
    lora_dropout=0.1,  # dropout probability
    r=64,  # lora dimension
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # training epochs
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=0,  # save every n of steps
    logging_steps=25,  # log every N steps
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",  # learning rate schedule
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model_name)

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model_name)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.push_to_hub(new_model_name, use_temp_dir=False)
tokenizer.push_to_hub(new_model_name, use_temp_dir=False)
