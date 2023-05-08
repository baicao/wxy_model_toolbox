import pandas as pd
import numpy as np
import datetime
import torch
from torch.utils.data import (
    TensorDataset,
    random_split,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)


def show_state(training_stats):
    # Display floats with two decimal places.
    pd.set_option("display.precision", 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index("epoch")

    # A hack to force the column headers to wrap.
    # df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    return df_stats


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def process_text(tokenizer, df):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    # For every sentence...
    ids = df.id.tolist()
    labels = df.label.tolist()
    for sent in df.text.tolist():
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=128,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict["attention_mask"])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    ids = torch.Tensor([int(id[:-1]) for id in ids])

    return input_ids, attention_masks, labels, ids


def split_train_test(input_ids, attention_masks, labels, ids):
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels, ids)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset


def dataset_2_dataloader(dataset, batch_size, is_random=False):
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    if is_random:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=sampler,  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
    )

    return dataloader
