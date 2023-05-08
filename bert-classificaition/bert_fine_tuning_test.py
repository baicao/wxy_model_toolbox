import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from bert_util import dataset_2_dataloader, process_text
from common_model.gpu_manager import GPUManager
from common.log_factory import logger
from common.classification_report import ClassificationReport

# 模型
# 数据文件
data_file = "data.csv"
model_name = "model_save"
df = pd.read_csv(data_file, sep="\001", header=0, dtype={"id": str})


# Report the number of sentences.
print("Number of test sentences: {:,}\n".format(df.shape[0]))

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
ids = df.id.tolist()

input_ids, attention_masks, labels, _ = process_text(tokenizer, df)


# Set the batch size.
batch_size = 32
try:
    gm = GPUManager()
    gpu_index = gm.auto_choice()
    logger.info("cuda index %s " % gpu_index)
    device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
except:
    device = "cpu"
logger.info("device %s" % device)

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_dataloader = dataset_2_dataloader(prediction_data, batch_size)
# Prediction on test set

print("Predicting labels for {:,} test sentences...".format(len(input_ids)))

# Put model in evaluation mode
model.eval()
model.to(device)

# Tracking variables
predictions, true_labels = [], []

# Predict
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    label_ids = b_labels.to("cpu").numpy()

    # Store predictions and true labels
    predictions.extend(pred)
    true_labels.extend(label_ids)

print("    DONE.")
print("ids", len(ids))
print("true_labels", len(true_labels))
print("predictions", len(predictions))

data = {"pred": predictions, "label": true_labels, "id": ids}
df = pd.DataFrame(data)
df.to_excel("predict.xlsx", index=False)
cr = ClassificationReport(labels=["受理范围", "非受理范围"], logger=logger)
cr.show_cm_report(true_labels, predictions)
