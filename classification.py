import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import time

def classify_batch(texts, classifier, candidate_labels):
    return [classifier(text, candidate_labels=candidate_labels) for text in texts]

def classify_texts(dataset, text_column, candidate_labels, model_name, batch_size=16):
    # Check if GPU is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    
    # Initialize the zero-shot classification pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    def preprocess_function(examples):
        return tokenizer(list(examples[text_column]), padding=True, truncation=True, max_length=512)

    # Ensure that the text column is correctly extracted
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Split dataset into batches
    batch_predictions = []
    for i in range(0, len(tokenized_dataset), batch_size):
        batch = tokenized_dataset.select(range(i, min(i + batch_size, len(tokenized_dataset))))
        texts = [example for example in batch[text_column]]
        batch_preds = classify_batch(texts, classifier, candidate_labels)
        batch_predictions.extend(batch_preds)

    # Extract the highest score label for each text
    highest_score_labels = [pred['labels'][pred['scores'].index(max(pred['scores']))] for pred in batch_predictions]

    return highest_score_labels

def classify_task(df, text_column, task_type):
    # Define candidate labels based on the task type and DataFrame language
    if df.equals(fbs_df) or df.equals(chinese_df):
        model_name = "joeddav/xlm-roberta-large-xnli"
        if task_type == "manipulation":
            candidate_labels = ["紧急", "威胁", "权威机构", "奖励", "冒充熟人"]
        elif task_type == "sentiment":
            candidate_labels = ["正面", "负面", "中性"]
        elif task_type == "topic":
            candidate_labels = ["金融机构", "政府", "交易", "快递", "冒充熟人"]
        else:
            raise ValueError("Invalid task type. Choose from 'manipulation', 'sentiment', or 'topic'.")
    else:
        model_name = "facebook/bart-large-mnli"
        if task_type == "manipulation":
            candidate_labels = ["urgency", "invoking fear", "authority", "incentives", "impersonation"]
        elif task_type == "sentiment":
            candidate_labels = ["positive", "negative", "neutral"]
        elif task_type == "topic":
            candidate_labels = ["finance", "government", "retail", "shipping", "impersonation of someone they know"]
        else:
            raise ValueError("Invalid task type. Choose from 'manipulation', 'sentiment', or 'topic'.")

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Make predictions and add results to the DataFrame
    df[f'{task_type}_label'] = classify_texts(dataset, text_column, candidate_labels, model_name=model_name)
    return df

# Load datasets
print("Load datasets")
fbs_df = pd.read_csv('processed_data/fbs_sms_df.csv')
chinese_df = pd.read_csv('processed_data/chinese_text_classification_df.csv')
mendeley_df = pd.read_csv('processed_data/mendeley_df.csv')
uci_df = pd.read_csv('processed_data/uci_df.csv')

# Classify each task and keep results in the original DataFrame
datasets = [("fbs_df", fbs_df), ("chinese_df", chinese_df), ("mendeley_df", mendeley_df), ("uci_df", uci_df)]
tasks = ['manipulation', 'sentiment', 'topic']

for name, df in datasets:
    for task in tasks:
        start_time = time.time()
        print(f"Processing {name} for {task}")
        df = classify_task(df, 'CLEANED_TEXT', task)
        duration = time.time() - start_time
        print(f"Completed {name} for {task} in {duration:.2f} seconds")
    df.to_csv(f"results_{name}.csv", index=False)

print("DONE")
