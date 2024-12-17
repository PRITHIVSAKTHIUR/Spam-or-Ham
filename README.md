---
title: Spam Detection
emoji: 🥷🏻
colorFrom: green
colorTo: pink
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Bert Uncased
---
### **Spam Detection with BERT**

This repository contains an implementation of a **Spam Detection** model using **BERT (Bidirectional Encoder Representations from Transformers)** for binary classification (Spam / Ham). The model is trained on the **`prithivMLmods/Spam-Text-Detect-Analysis` dataset** and leverages **Weights & Biases (wandb)** for comprehensive experiment tracking.

---
## **🗂️ Summary of Uploaded Files**

| **File Name**                     | **Size**  | **Description**                                     | **Upload Status** |
|------------------------------------|-----------|-----------------------------------------------------|-------------------|
| `.gitattributes`                   | 1.52 kB   | Tracks files stored with Git LFS.                  | Uploaded          |
| `README.md`                        | 8.78 kB   | Comprehensive documentation for the repository.    | Updated           |
| `config.json`                      | 727 Bytes | Configuration file related to the model settings.   | Uploaded          |
| `model.safetensors`                | 438 MB   | Model weights stored in safetensors format.        | Uploaded (LFS)    |
| `special_tokens_map.json`          | 125 Bytes | Mapping of special tokens for tokenizer handling.  | Uploaded          |
| `tokenizer_config.json`            | 1.24 kB   | Tokenizer settings for initialization.              | Uploaded          |
| `vocab.txt`                         | 232 kB   | Vocabulary file for tokenizer use.                 | Uploaded          |

---
## **🛠️ Overview**

### **Core Details:**
- **Model:** BERT for sequence classification  
  Pre-trained Model: `bert-base-uncased`
- **Task:** Spam detection - Binary classification task (Spam vs Ham).
- **Metrics Tracked:**  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - Evaluation loss

---

## **📊 Key Results**
Results were obtained using BERT and the provided training dataset:

- **Validation Accuracy:** **0.9937**  
- **Precision:** **0.9931**  
- **Recall:** **0.9597**  
- **F1 Score:** **0.9761**
---
## **📈 Model Training Details**

### **Model Architecture:**
The model uses `bert-base-uncased` as the pre-trained backbone and is fine-tuned for the sequence classification task.

### **Training Parameters:**
- **Learning Rate:** 2e-5  
- **Batch Size:** 16  
- **Epochs:** 3  
- **Loss:** Cross-Entropy
---
## **🚀 How to Use the Model**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd <project-directory>
```

### **2. Install Dependencies**
Install all necessary dependencies.
```bash
pip install -r requirements.txt
```
or manually:
```bash
pip install transformers datasets wandb scikit-learn
```
### **3. Train the Model**
Assuming you have a script like `train.py`, run:
```python
# Import necessary libraries
from datasets import load_dataset, ClassLabel
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
dataset = load_dataset("prithivMLmods/Spam-Text-Detect-Analysis", split="train")

# Encode labels as integers
label_mapping = {"ham": 0, "spam": 1}
dataset = dataset.map(lambda x: {"label": label_mapping[x["Category"]]})
dataset = dataset.rename_column("Message", "text").remove_columns(["Category"])

# Convert label column to ClassLabel for stratification
class_label = ClassLabel(names=["ham", "spam"])
dataset = dataset.cast_column("label", class_label)

# Split into train and test
dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define evaluation metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate after every epoch
    save_strategy="epoch",        # Save checkpoint after every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)

# Save the trained model
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# Load the model for inference
loaded_model = BertForSequenceClassification.from_pretrained("./saved_model").to(device)
loaded_tokenizer = BertTokenizer.from_pretrained("./saved_model")

# Test the model on a custom input
def predict(text):
    inputs = loaded_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device as model
    outputs = loaded_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Spam" if prediction == 1 else "Ham"

# Example test
example_text = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now."
print("Prediction:", predict(example_text))
```
---
## **🔗 Dataset Information**
The training dataset comes from **Spam-Text-Detect-Analysis** available on Hugging Face:
- **Dataset Link:** [Spam Text Detection Dataset - Hugging Face](https://huggingface.co/datasets)

Dataset size:
- **5.57k entries**
---
## **✨ Weights & Biases Integration**

### Why Use wandb?
- **Monitor experiments in real time** via visualization.
- Log metrics such as loss, accuracy, precision, recall, and F1 score.
- Provides a history of past runs and their comparisons.

### Initialize Weights & Biases
Include this snippet in your training script:
```python
import wandb
wandb.init(project="spam-detection")
```
---
## **📁 Directory Structure**

The directory is organized to ensure scalability and clear separation of components:

```
project-directory/
│
├── data/                # Dataset processing scripts
├── wandb/              # Logged artifacts from wandb runs
├── results/            # Save training and evaluation results
├── model/              # Trained model checkpoints
├── requirements.txt    # List of dependencies
└── train.py            # Main script for training the model
```
---
## **🌐 Gradio Interface**

A Gradio interface is provided to test the model interactively. The interface allows users to input text and get predictions on whether the text is **Spam** or **Ham**.

### **Example Usage**
```python
import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained BERT model and tokenizer
MODEL_PATH = "prithivMLmods/Spam-Bert-Uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Function to predict if a given text is Spam or Ham
def predict_spam(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, axis=-1).item()
    return "Spam" if prediction == 1 else "Ham"

# Gradio UI
inputs = gr.Textbox(label="Enter Text", placeholder="Type a message to check if it's Spam or Ham...")
outputs = gr.Label(label="Prediction")

examples = [
    ["Win $1000 gift cards now by clicking here!"],
    ["You have been selected for a lottery."],
    ["Hello, how was your day?"],
    ["Earn money without any effort. Click here."],
    ["Meeting tomorrow at 10 AM. Don't be late."],
    ["Claim your free prize now!"],
    ["Are we still on for dinner tonight?"],
    ["Exclusive offer just for you, act now!"],
    ["Let's catch up over coffee soon."],
    ["Congratulations, you've won a new car!"]
]

gr_interface = gr.Interface(
    fn=predict_spam,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title="Spam Detection with BERT",
    description="Type a message in the text box to check if it's Spam or Ham using a pre-trained BERT model."
)

gr_interface.launch()
```
---
