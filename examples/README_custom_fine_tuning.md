# Custom Dataset Fine-tuning Example with Tetra

This example demonstrates how to use Tetra to fine-tune a pre-trained language model on a custom dataset. The example shows how to:

1. Create a custom dataset from a CSV file
2. Upload the dataset to a remote server
3. Fine-tune a model on the custom dataset
4. Use the fine-tuned model for predictions

## Prerequisites

Before running this example, make sure you have:

1. Set up your Tetra environment
2. Created a `.env` file with your RunPod API key:
   ```
   RUNPOD_API_KEY=your_runpod_api_key
   ```
3. Installed the required dependencies:
   ```
   pip install tetra dotenv pandas
   ```

## How It Works

The example consists of four main functions:

1. `create_custom_dataset()`: Creates a dataset from a CSV file and splits it into training and validation sets.
2. `upload_dataset()`: Uploads the dataset to the remote server.
3. `fine_tune_custom_model()`: Fine-tunes a pre-trained model on the custom dataset.
4. `classify_text()`: Uses the fine-tuned model to classify new text.

### Creating a Custom Dataset

The example shows how to create a custom dataset from a CSV file. The CSV file should have at least two columns:
- A text column containing the input text
- A label column containing the classification labels

The function splits the dataset into training and validation sets (80/20 split) and saves it as a JSON file.

### Uploading the Dataset

The dataset is uploaded to the remote server using the `upload_dataset()` function. This function:
1. Creates a directory on the remote server to store the dataset
2. Saves the dataset JSON to the remote server
3. Creates Hugging Face Dataset objects for training and validation
4. Saves the datasets to disk for later use

### Fine-tuning the Model

The `fine_tune_custom_model()` function fine-tunes a pre-trained model on the custom dataset. It:
1. Loads the dataset from disk
2. Maps string labels to integers if needed
3. Loads the pre-trained model and tokenizer
4. Tokenizes the dataset
5. Sets up training arguments
6. Trains the model
7. Evaluates the model
8. Saves the model and label mapping to disk

### Making Predictions

The `classify_text()` function uses the fine-tuned model to classify new text. It:
1. Loads the model, tokenizer, and label mapping
2. Tokenizes the input text
3. Makes a prediction
4. Maps the predicted class ID back to the original label
5. Returns the predicted label, confidence, and class probabilities

## Example Dataset

The example includes a small product review dataset with three classes:
- positive
- negative
- neutral

You can replace this with your own dataset by modifying the CSV file or creating a new one.

## Running the Example

To run the example:

```bash
python examples/custom_dataset_fine_tuning.py
```

The script will:
1. Create a custom dataset from the example data
2. Upload the dataset to the remote server
3. Fine-tune the model on the custom dataset
4. Make predictions on example texts
5. Print the results

## GPU Resource Configuration

The example uses a GPU resource with the following configuration:

```python
gpu_config = {
    "api_key": os.environ.get("RUNPOD_API_KEY"),
    "template_id": "jizsa65yn0",  # Replace with your template ID
    "gpu_ids": "AMPERE_48",
    "workers_min": 1,
    "workers_max": 1,
    "name": "custom-fine-tuning-server",
}
```

Make sure to replace the `template_id` with your own RunPod template ID.

## Customizing the Example

You can customize the example by:
- Using your own CSV file with different text and labels
- Changing the model architecture (e.g., using "bert-base-uncased" instead of "distilbert-base-uncased")
- Adjusting the training parameters (epochs, batch size, learning rate)
- Modifying the tokenization parameters (max length)

## Multi-class vs. Binary Classification

The example handles both binary and multi-class classification automatically. It detects the number of unique labels in your dataset and configures the model accordingly. 