# ✅ STEP 1: Install required libraries (run in terminal if not already)
# pip install transformers datasets peft accelerate bitsandbytes tqdm

# ✅ STEP 2: Load and prepare dataset
from datasets import load_dataset

dataset_path = "training-data/indian_cooking.jsonl"
dataset = load_dataset("json", data_files=dataset_path)
dataset = dataset["train"].train_test_split(test_size=0.1)

# ✅ STEP 3: Load model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch

model_name = "microsoft/phi-1_5"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding token error

# Load model on CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# ✅ STEP 4: Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# ✅ STEP 5: Format prompt and tokenize (FIXED)
def format_prompt(prompt, response):
    return f"### Instruction:\n{prompt}\n\n### Response:\n{response}"

def tokenize(batch):
    prompts = [format_prompt(p, r) for p, r in zip(batch["prompt"], batch["response"])]
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return {k: v.tolist() for k, v in tokenized.items()}

tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names
)
tokenized_dataset = tokenized_dataset.with_format("torch")

# ✅ STEP 6: Define training arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./phi1_5-cooking-lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    no_cuda=True,
    disable_tqdm=False  # Show training progress
)

# ✅ STEP 7: Data Collator
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked
)

# ✅ STEP 8: Trainer setup
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ✅ STEP 9: Start training
print("🚀 Starting fine-tuning...")
trainer.train()
print("✅ Training complete!")

# ✅ STEP 10: Save the model
save_path = "./phi1_5-cooking-lora"
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"📁 Model saved to: {save_path}")
