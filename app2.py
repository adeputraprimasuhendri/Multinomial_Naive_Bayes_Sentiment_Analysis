import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import pickle

print("Loading dataset...")
# ============================
# 1. Load Dataset from CSV
# ============================
df = pd.read_csv('clean.csv')
df.columns = ['text', 'label']

# Map labels to numeric values
label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
label_map_reverse = {2: 'positive', 1: 'neutral', 0: 'negative'}
df['label'] = df['label'].map(label_map)

print(f"Dataset loaded: {len(df)} records")
print(f"Label distribution:\n{df['label'].value_counts()}")

# ============================
# 2. Train-Test Split
# ============================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")

# ============================
# 3. Load IndoBERT Model and Tokenizer
# ============================
print("\nLoading IndoBERT model and tokenizer...")
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,  # positive, neutral, negative
    id2label=label_map_reverse,
    label2id=label_map
)

print(f"Model loaded: {model_name}")

# ============================
# 4. Tokenize Dataset
# ============================
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# ============================
# 5. Training Configuration
# ============================
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

# ============================
# 6. Define Metrics
# ============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

# ============================
# 7. Initialize Trainer
# ============================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ============================
# 8. Train Model
# ============================
print("\nStarting training...")
trainer.train()

# ============================
# 9. Evaluate Model
# ============================
print("\nEvaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# ============================
# 10. Save Model and Tokenizer
# ============================
print("\nSaving model and tokenizer...")
model.save_pretrained("./indobert_sentiment_model")
tokenizer.save_pretrained("./indobert_sentiment_model")

# Save label mapping
with open('label_map.pkl', 'wb') as f:
    pickle.dump({'label_map': label_map, 'label_map_reverse': label_map_reverse}, f)

print("Model saved successfully to './indobert_sentiment_model'")

# ============================
# 11. Test Predictions
# ============================
print("\n=== Testing with new data ===")

# Load saved model
loaded_model = AutoModelForSequenceClassification.from_pretrained("./indobert_sentiment_model")
loaded_tokenizer = AutoTokenizer.from_pretrained("./indobert_sentiment_model")

# Test data
data_baru = [
    "Produk ini sangat mengecewakan",
    "Saya puas dengan kualitas barangnya",
    "Barang ini biasa saja menurut saya",
    "Tidak sesuai keinginan",
    "Pengiriman cepat dan memuaskan"
]

print("\nPredicting sentiments...")
predictions = []

for text in data_baru:
    inputs = loaded_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_label = label_map_reverse[predicted_class]
        predictions.append(predicted_label)

# Display results
df_hasil = pd.DataFrame({
    "Ulasan Baru": data_baru,
    "Prediksi Sentimen": predictions
})

print("\n=== Hasil Prediksi Data Testing ===")
print(df_hasil)

# ============================
# 12. Detailed Evaluation on Test Set
# ============================
print("\n=== Detailed Test Set Evaluation ===")
test_predictions = trainer.predict(test_dataset)
y_pred = np.argmax(test_predictions.predictions, axis=-1)
y_true = test_predictions.label_ids

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=['negative', 'neutral', 'positive']
))

print(f"\nTest Accuracy: {accuracy_score(y_true, y_pred):.4f}")
