from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os

# Step 1: Load PubMedQA dataset
dataset = load_dataset("pubmed_qa", "pqa_labeled")

# Step 2: Manually split train into train and validation
train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Step 3: Load T5 model and tokenizer
model_ckpt = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to("cpu")  # Use CPU for resource efficiency

# Step 4: Preprocess the dataset
def preprocess_function(examples):
    """
    Preprocess batched inputs for tokenization.
    """
    # Extract text from "context" (if it's a dict or list of dicts)
    context_texts = [
        c["text"] if isinstance(c, dict) and "text" in c else c if isinstance(c, str) else ""
        for c in examples["context"]
    ]

    # Combine question and context for each example
    inputs = tokenizer(
        [q + " " + c for q, c in zip(examples["question"], context_texts)],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Tokenize the long answer as the target
    targets = tokenizer(
        examples["long_answer"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=16, num_proc=1)
eval_dataset = eval_dataset.map(preprocess_function, batched=True, batch_size=16, num_proc=1)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
)

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# Step 7: Train the model
trainer.train()

# Step 8: Save the fine-tuned model
output_dir = "./fine_tuned_pubmedqa_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine-tuned T5 model saved to: {output_dir}")
