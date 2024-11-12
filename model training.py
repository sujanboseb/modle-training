# Install required libraries
!pip install transformers datasets accelerate

# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# Load the dataset from CSV file
import pandas as pd

# Load your CSV file
csv_file_path = '’Content/MyDrive/intent_dataset(1).csv'  # Update this with the actual path to your CSV
df = pd.read_csv(csv_file_path)

# Ensure your CSV has 'input' and 'intent' columns
df_train, df_eval = train_test_split(df, test_size=0.1)

# Convert DataFrame to Dataset
train_dataset = Dataset.from_pandas(df_train)
eval_dataset = Dataset.from_pandas(df_eval)

# Load the tokenizer and model
model_name = "google/flan-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Preprocessing the data
def preprocess_function(examples):
    inputs = [ex for ex in examples['input']]
    targets = [ex for ex in examples['intent']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length').input_ids

    model_inputs["labels"] = labels
    return model_inputs

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./flan-large-finetuned",  # Where to save the model
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('/content/drive/MyDrive/flan-large-finetuned')
tokenizer.save_pretrained('/content/drive/MyDrive/flan-large-finetuned')


