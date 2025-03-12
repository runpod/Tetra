import asyncio
import os
from dotenv import load_dotenv
from tetra import remote, get_global_client

# Load environment variables from .env file
load_dotenv()

# Configuration for a GPU resource
gpu_config = {
    "api_key": os.environ.get("RUNPOD_API_KEY"),
    "template_id": "jizsa65yn0",  # Replace with your template ID
    "gpu_ids": "AMPERE_48",
    "workers_min": 1,  # Keep worker alive for persistence
    "workers_max": 1,
    "name": "fine-tuning-server",
}


# Initialize and fine-tune a model
@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=[
        "torch==2.0.1", 
        "transformers==4.30.2", 
        "datasets==2.13.1", 
        "accelerate==0.20.3",
        "evaluate==0.4.0",
        "scikit-learn==1.2.2"
    ],
)
def fine_tune_model(
    model_name="distilbert-base-uncased",
    dataset_name="imdb",
    num_train_epochs=3,
    batch_size=16,
    learning_rate=5e-5,
):
    """Fine-tune a transformer model on a text classification dataset."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from datasets import load_dataset
    import evaluate
    import numpy as np
    import os
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Set up model directory for persistence
    model_dir = "/tmp/fine_tuned_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if model is already fine-tuned
    if os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"Loading already fine-tuned model from {model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return {
            "status": "loaded",
            "model_path": model_dir,
            "model_name": model_name,
            "dataset": dataset_name,
        }
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    try:
        # Try loading the dataset directly
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset directly: {str(e)}")
        print("Trying alternative loading method...")
        
        # Alternative loading method for IMDB dataset
        if dataset_name == "imdb":
            try:
                # Try loading with specific configuration
                dataset = load_dataset("imdb", ignore_verifications=True)
            except Exception as e2:
                print(f"Error with alternative loading method: {str(e2)}")
                
                # Manual creation of a small IMDB-like dataset as fallback
                print("Creating a small synthetic dataset as fallback...")
                
                # Create a small synthetic dataset
                positive_texts = [
                    "This movie was fantastic! I loved every minute of it.",
                    "One of the best films I've seen in years. Highly recommended!",
                    "Great acting, compelling story, and beautiful cinematography.",
                    "A masterpiece that will be remembered for generations.",
                    "I was completely captivated from beginning to end."
                ]
                
                negative_texts = [
                    "This movie was terrible. Complete waste of time.",
                    "I couldn't even finish watching it, that's how bad it was.",
                    "Poor acting, predictable plot, and awful direction.",
                    "One of the worst films I've seen in recent memory.",
                    "I want my money back after watching this disaster."
                ]
                
                train_texts = positive_texts + negative_texts
                train_labels = [1] * len(positive_texts) + [0] * len(negative_texts)
                
                # Double the data for test set (just for demonstration)
                test_texts = train_texts + train_texts
                test_labels = train_labels + train_labels
                
                # Create dataset dictionary
                dataset = {
                    "train": {"text": train_texts, "label": train_labels},
                    "test": {"text": test_texts, "label": test_labels}
                }
                
                # Convert to Dataset objects
                from datasets import Dataset
                dataset = {
                    "train": Dataset.from_dict(dataset["train"]),
                    "test": Dataset.from_dict(dataset["test"])
                }
        else:
            raise ValueError(f"Could not load dataset: {dataset_name}")
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # Binary classification for IMDB
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    # Handle both HF Dataset and dictionary of datasets
    if hasattr(dataset, "map"):
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        train_dataset = tokenized_datasets["train"].shuffle(seed=42)
        eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
    else:
        tokenized_train = dataset["train"].map(tokenize_function, batched=True)
        tokenized_test = dataset["test"].map(tokenize_function, batched=True)
        train_dataset = tokenized_train.shuffle(seed=42)
        eval_dataset = tokenized_test.shuffle(seed=42)
    
    # Define metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("Starting fine-tuning...")
    train_result = trainer.train()
    
    # Evaluate model
    eval_result = trainer.evaluate()
    
    # Save model
    print(f"Saving fine-tuned model to {model_dir}")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Return results
    return {
        "status": "fine-tuned",
        "model_path": model_dir,
        "model_name": model_name,
        "dataset": dataset_name,
        "training_metrics": {
            "train_loss": float(train_result.training_loss),
            "train_runtime": train_result.metrics["train_runtime"],
        },
        "eval_metrics": {
            "accuracy": float(eval_result["eval_accuracy"]),
            "f1": float(eval_result["eval_f1"]),
            "precision": float(eval_result["eval_precision"]),
            "recall": float(eval_result["eval_recall"]),
        },
    }


# Evaluate the fine-tuned model on new text
@remote(resource_config=gpu_config, resource_type="serverless")
def predict_sentiment(text):
    """Use the fine-tuned model to predict sentiment of a text."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import os
    
    model_dir = "/tmp/fine_tuned_model"
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        return {"error": "Model not fine-tuned. Call fine_tune_model first."}
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    # Get predicted class and confidence
    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = predictions[0][predicted_class].item()
    
    # Map class to sentiment (for IMDB: 0 = negative, 1 = positive)
    sentiment = "positive" if predicted_class == 1 else "negative"
    
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "class_probabilities": {
            "negative": float(predictions[0][0]),
            "positive": float(predictions[0][1]),
        }
    }


async def main():
    # Step 1: Fine-tune the model
    print("Starting fine-tuning process...")
    fine_tune_result = await fine_tune_model(
        model_name="distilbert-base-uncased",
        dataset_name="imdb",
        num_train_epochs=1,  # Using just 1 epoch for demonstration
        batch_size=8,
        learning_rate=2e-5,
    )
    print(f"Fine-tuning result: {fine_tune_result}")
    
    # Step 2: Make predictions with the fine-tuned model
    print("\nMaking predictions with the fine-tuned model...")
    
    # Positive example
    positive_text = "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
    positive_result = await predict_sentiment(positive_text)
    print(f"Positive example result: {positive_result}")
    
    # Negative example
    negative_text = "What a terrible waste of time. The plot made no sense and the acting was wooden."
    negative_result = await predict_sentiment(negative_text)
    print(f"Negative example result: {negative_result}")


if __name__ == "__main__":
    asyncio.run(main()) 