import os
import logging
import requests
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import torch
from seqeval.metrics import classification_report as seqeval_classification_report
import wandb
from transformers.integrations import WandbCallback

# Set environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure WandB login
keyFile = open('wandb.key', 'r')
WANDB_API_KEY = keyFile.readline().rstrip()
wandb.login(key=WANDB_API_KEY)

# URLs to the IOB datasets
train_url = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/SETH-train.iob'
test_url = 'https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/IOB/SETH-test.iob'

# Function to download and parse the dataset
def load_and_parse_iob(url):
    response = requests.get(url)
    data = response.text.split('\n')

    sentences = []
    labels = []
    sentence = []
    label = []

    for line in data:
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            if sentence:
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
        else:
            parts = line.split(',')
            if len(parts) == 2:
                token, tag = parts
                token, tag = token.strip(), tag.strip()
                sentence.append(token)
                label.append(tag)

    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels

# Load and parse the training and test datasets
train_sentences, train_labels = load_and_parse_iob(train_url)
test_sentences, test_labels = load_and_parse_iob(test_url)

# Convert to DataFrame
train_df = pd.DataFrame({'sentence': train_sentences, 'labels': train_labels})
test_df = pd.DataFrame({'sentence': test_sentences, 'labels': test_labels})

# Verify unique labels before cleaning
unique_labels = list(set(label for sublist in train_labels for label in sublist))

# Define the expected labels
expected_labels = {'O', 'B-Gene', 'I-SNP', 'I-Gene', 'B-SNP', 'B-RS'}

# Function to clean labels
def clean_labels(label_list):
    return [label if label in expected_labels else 'O' for label in label_list]

# Clean the labels
cleaned_train_labels = [clean_labels(label_list) for label_list in train_labels]
cleaned_test_labels = [clean_labels(label_list) for label_list in test_labels]

# Update the DataFrames with cleaned labels
train_df['labels'] = cleaned_train_labels
test_df['labels'] = cleaned_test_labels

# Verify the cleaned labels
cleaned_unique_labels = list(set(label for sublist in cleaned_train_labels for label in sublist))

# Define the tokenizer and model checkpoint
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Create Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Split training dataset into train and validation datasets
dataset = train_dataset.train_test_split(test_size=0.1)

# Function to tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, padding=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(cleaned_unique_labels.index(label[word_idx]))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize the datasets
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["sentence", "labels"])
tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=["sentence", "labels"])

# Define training function for the sweep
def train():
    run = wandb.init()
    config = run.config
    
    try:
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            report_to="wandb"  # Integrate with WandB
        )

        # Data collator to handle dynamic padding
        data_collator = DataCollatorForTokenClassification(tokenizer)

        # Initialize the Trainer
        trainer = Trainer(
            model=AutoModelForTokenClassification.from_pretrained(config.model_checkpoint, num_labels=len(cleaned_unique_labels)),
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[WandbCallback()]  # Include WandbCallback for better integration
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        trainer.evaluate()

        # Function to convert predictions to labels
        def align_predictions(predictions, label_ids):
            preds = predictions.argmax(-1)
            batch_size, seq_len = preds.shape

            out_label_list = [[] for _ in range(batch_size)]
            preds_list = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                for j in range(seq_len):
                    if label_ids[i, j] != -100:
                        out_label_list[i].append(cleaned_unique_labels[label_ids[i][j]])
                        preds_list[i].append(cleaned_unique_labels[preds[i][j]])

            return preds_list, out_label_list

        # Evaluate on the test set
        test_predictions = trainer.predict(tokenized_test_dataset)
        preds_list, out_label_list = align_predictions(test_predictions.predictions, test_predictions.label_ids)
        report = seqeval_classification_report(out_label_list, preds_list)
        logger.info(report)
        wandb.log({"classification_report": report})
        # Log the metrics to W&B
        wandb.log({
            "final_learning_rate": config.learning_rate,
            "model_checkpoint": config.model_checkpoint,
        })
    except Exception as e:
        logger.error(f"Error during training or evaluation: {e}")
        wandb.alert(
            title="Training or Evaluation Error",
            text=f"An error occurred during training or evaluation: {e}",
            level=wandb.AlertLevel.ERROR
        )

    finally:
        run.finish()

# Define the Sweep Configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'eval_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.00001,
            'max': 0.01
        },
        'model_checkpoint': {
            'values': ["bert-base-cased", "bert-large-cased", "roberta-base", "roberta-large"]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="ner-sweep5")

# Run the sweep
wandb.agent(sweep_id, function=train, count=3)
