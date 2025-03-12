# Fine-tuning Example with Tetra

This example demonstrates how to use Tetra to fine-tune a pre-trained language model on a text classification task. The example uses the IMDB dataset to fine-tune a DistilBERT model for sentiment analysis.

## Prerequisites

Before running this example, make sure you have:

1. Set up your Tetra environment
2. Created a `.env` file with your RunPod API key:
   ```
   RUNPOD_API_KEY=your_runpod_api_key
   ```
3. Installed the required dependencies:
   ```
   pip install tetra dotenv
   ```

## How It Works

The example consists of two main functions:

1. `fine_tune_model()`: This function fine-tunes a pre-trained model on a dataset.
2. `predict_sentiment()`: This function uses the fine-tuned model to predict sentiment on new text.

Both functions are decorated with the `@remote` decorator, which means they will be executed on a remote GPU server provisioned by Tetra.

### Fine-tuning Process

The fine-tuning process follows these steps:

1. Load the dataset (IMDB by default)
2. Load the pre-trained model and tokenizer (DistilBERT by default)
3. Tokenize the dataset
4. Set up training arguments
5. Train the model
6. Evaluate the model
7. Save the model to a persistent location

The fine-tuned model is saved to `/tmp/fine_tuned_model` on the remote server, which persists between calls as long as the server is running.

### Making Predictions

The prediction function:

1. Loads the fine-tuned model and tokenizer
2. Tokenizes the input text
3. Makes a prediction
4. Returns the predicted sentiment (positive or negative) along with confidence scores

## Configuration

You can customize the fine-tuning process by modifying these parameters:

- `model_name`: The pre-trained model to use (default: "distilbert-base-uncased")
- `dataset_name`: The dataset to fine-tune on (default: "imdb")
- `num_train_epochs`: Number of training epochs (default: 3)
- `batch_size`: Batch size for training (default: 16)
- `learning_rate`: Learning rate for training (default: 5e-5)

## Running the Example

To run the example:

```bash
python fine_tuning.py
```

The script will:
1. Fine-tune the model (or load an already fine-tuned model if it exists)
2. Make predictions on example texts
3. Print the results

## GPU Resource Configuration

The example uses a GPU resource with the following configuration:

```python
gpu_config = {
    "api_key": os.environ.get("RUNPOD_API_KEY"),
    "template_id": "jizsa65yn0",  # Replace with your template ID
    "gpu_ids": "AMPERE_48",
    "workers_min": 1,
    "workers_max": 1,
    "name": "fine-tuning-server",
}
```

Make sure to replace the `template_id` with your own RunPod template ID.

## Dependencies

The example requires the following Python packages on the remote server:

- torch
- transformers
- datasets
- accelerate
- evaluate
- scikit-learn

These are automatically installed on the remote server when the function is executed. 