import asyncio
import os
import json
import pandas as pd
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
    "name": "custom-fine-tuning-server",
}


# Create a custom dataset from a CSV file
def create_custom_dataset(csv_file, text_column, label_column):
    """Create a custom dataset from a CSV file."""
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Create a dataset dictionary
    dataset = {
        "train": {
            "text": df[text_column].tolist(),
            "label": df[label_column].tolist(),
        }
    }
    
    # Split into train and validation sets (80/20 split)
    train_size = int(0.8 * len(df))
    
    dataset["train"] = {
        "text": df[text_column][:train_size].tolist(),
        "label": df[label_column][:train_size].tolist(),
    }
    
    dataset["validation"] = {
        "text": df[text_column][train_size:].tolist(),
        "label": df[label_column][train_size:].tolist(),
    }
    
    # Save the dataset to a JSON file
    with open("custom_dataset.json", "w") as f:
        json.dump(dataset, f)
    
    return "custom_dataset.json"


# Upload a dataset to the remote server
@remote(
    resource_config=gpu_config,
    resource_type="serverless",
    dependencies=["torch", "transformers", "datasets", "pandas", "numpy"],
)
def upload_dataset(dataset_json):
    """Upload a dataset to the remote server."""
    import json
    import os
    from datasets import Dataset
    
    # Create directory for dataset
    dataset_dir = "/tmp/custom_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save the dataset JSON to the remote server
    dataset_path = os.path.join(dataset_dir, "dataset.json")
    with open(dataset_path, "w") as f:
        f.write(dataset_json)
    
    # Load the dataset
    dataset_dict = json.loads(dataset_json)
    
    # Create Hugging Face datasets
    train_dataset = Dataset.from_dict(dataset_dict["train"])
    validation_dataset = Dataset.from_dict(dataset_dict["validation"])
    
    # Save datasets
    train_dataset.save_to_disk(os.path.join(dataset_dir, "train"))
    validation_dataset.save_to_disk(os.path.join(dataset_dir, "validation"))
    
    # Get label set
    unique_labels = set(dataset_dict["train"]["label"])
    
    return {
        "status": "uploaded",
        "dataset_path": dataset_dir,
        "num_train_examples": len(dataset_dict["train"]["label"]),
        "num_validation_examples": len(dataset_dict["validation"]["label"]),
        "unique_labels": list(unique_labels),
        "num_labels": len(unique_labels),
    }


# Fine-tune a model on a custom dataset
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
def fine_tune_custom_model(
    model_name="distilbert-base-uncased",
    num_train_epochs=3,
    batch_size=16,
    learning_rate=5e-5,
    max_length=128,
):
    """Fine-tune a transformer model on a custom dataset."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from datasets import load_from_disk
    import numpy as np
    import os
    import json
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Set up model directory for persistence
    model_dir = "/tmp/custom_fine_tuned_model"
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
        }
    
    # Load dataset
    dataset_dir = "/tmp/custom_dataset"
    if not os.path.exists(dataset_dir):
        return {"error": "Dataset not uploaded. Call upload_dataset first."}
    
    print("Loading custom dataset...")
    train_dataset = load_from_disk(os.path.join(dataset_dir, "train"))
    validation_dataset = load_from_disk(os.path.join(dataset_dir, "validation"))
    
    # Get number of labels
    with open(os.path.join(dataset_dir, "dataset.json"), "r") as f:
        dataset_dict = json.load(f)
    
    unique_labels = set(dataset_dict["train"]["label"])
    num_labels = len(unique_labels)
    label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
    
    # Map string labels to integers if needed
    if not all(isinstance(label, int) for label in dataset_dict["train"]["label"]):
        train_dataset = train_dataset.map(
            lambda x: {"label": label_map[x["label"]]}, 
            remove_columns=None
        )
        validation_dataset = validation_dataset.map(
            lambda x: {"label": label_map[x["label"]]}, 
            remove_columns=None
        )
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_validation = validation_dataset.map(tokenize_function, batched=True)
    
    # Define metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Handle binary or multi-class classification
        if num_labels == 2:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average="binary"
            )
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average="weighted"
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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
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
    
    # Save label mapping
    with open(os.path.join(model_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)
    
    # Return results
    return {
        "status": "fine-tuned",
        "model_path": model_dir,
        "model_name": model_name,
        "num_labels": num_labels,
        "label_map": label_map,
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


# Classify text with the fine-tuned model
@remote(resource_config=gpu_config, resource_type="serverless")
def classify_text(text):
    """Use the fine-tuned model to classify text."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import os
    import json
    import traceback
    
    try:
        model_dir = "/tmp/custom_fine_tuned_model"
        
        # Check if model exists
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            return {"error": "Model not fine-tuned. Call fine_tune_custom_model first."}
        
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load label mapping
        label_map_path = os.path.join(model_dir, "label_map.json")
        if not os.path.exists(label_map_path):
            print(f"Label map not found at {label_map_path}, creating default mapping")
            # Create a default mapping based on the number of labels
            num_labels = model.config.num_labels
            label_map = {str(i): i for i in range(num_labels)}
            id_to_label = {i: str(i) for i in range(num_labels)}
        else:
            with open(label_map_path, "r") as f:
                label_map = json.load(f)
                
            # Convert keys to strings if they aren't already
            if not all(isinstance(k, str) for k in label_map.keys()):
                label_map = {str(k): v for k, v in label_map.items()}
                
            # Create reverse mapping
            id_to_label = {v: k for k, v in label_map.items()}
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get predicted class and confidence
        predicted_class_id = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class_id].item()
        
        # Map class ID to label
        if str(predicted_class_id) not in id_to_label:
            print(f"Warning: Predicted class ID {predicted_class_id} not found in mapping")
            predicted_label = str(predicted_class_id)
        else:
            predicted_label = id_to_label[str(predicted_class_id)]
        
        # Create class probabilities dict
        class_probs = {}
        for i in range(len(predictions[0])):
            if str(i) in id_to_label:
                class_probs[id_to_label[str(i)]] = float(predictions[0][i])
            else:
                class_probs[f"class_{i}"] = float(predictions[0][i])
        
        return {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "class_probabilities": class_probs
        }
    except Exception as e:
        error_msg = f"Error in classify_text: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}


async def main():
    # Example custom dataset (product reviews)
    custom_data = """text,label
"This product is amazing! I love it.",positive
"Worst purchase ever. Don't buy this.",negative
"It's okay, but not worth the price.",neutral
"Great quality and fast shipping.",positive
"Broke after one week of use.",negative
"Decent product for the price.",neutral
"Absolutely love this product! Will buy again.",positive
"Terrible customer service and product quality.",negative
"It works as expected, nothing special.",neutral
"Best purchase I've made this year!",positive
"Complete waste of money.",negative
"It's fine, does what it's supposed to do.",neutral
"Exceeded my expectations in every way.",positive
"Disappointed with the quality.",negative
"Average product, average performance.",neutral
"""
    
    # Save to CSV
    with open("product_reviews.csv", "w") as f:
        f.write(custom_data)
    
    # Step 1: Create and upload the custom dataset
    print("Creating custom dataset...")
    dataset_file = create_custom_dataset(
        "product_reviews.csv", 
        text_column="text", 
        label_column="label"
    )
    
    # Read the dataset JSON
    with open(dataset_file, "r") as f:
        dataset_json = f.read()
    
    print("Uploading dataset to remote server...")
    upload_result = await upload_dataset(dataset_json)
    print(f"Upload result: {upload_result}")
    
    # Step 2: Fine-tune the model on the custom dataset
    print("\nStarting fine-tuning process...")
    fine_tune_result = await fine_tune_custom_model(
        model_name="distilbert-base-uncased",
        num_train_epochs=3,
        batch_size=4,
        learning_rate=2e-5,
        max_length=128,
    )
    print(f"Fine-tuning result: {fine_tune_result}")
    
    # Step 3: Make predictions with the fine-tuned model
    print("\nMaking predictions with the fine-tuned model...")
    
    # Test examples
    test_texts = [
        "This is an excellent product, I'm very satisfied with my purchase.",
        "I regret buying this, it's of poor quality and overpriced.",
        "The product is adequate for basic needs, but nothing special."
    ]
    
    for text in test_texts:
        result = await classify_text(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['predicted_label']} (Confidence: {result['confidence']:.4f})")
        print(f"Class probabilities: {result['class_probabilities']}")


if __name__ == "__main__":
    asyncio.run(main()) 