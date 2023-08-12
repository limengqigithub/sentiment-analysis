import os
import sys

import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torch.optim import AdamW
from tqdm import tqdm, trange
import time
import numpy as np
from sklearn import preprocessing
from sklearn import metrics

# Load Pickle Paths
# dataset = "stackoverflow"
# train = f"./train_{dataset}.csv"
# test = f"./test_{dataset}.csv"
train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]
dataset_name = sys.argv[4]
# initialize roberta model
cur_model = (RobertaForSequenceClassification, RobertaTokenizer, './RoBERTa/roberta-base')
m_name = "Roberta"
# CHANGE ME
dataset = "stackoverflow"
splitnumber = "3"

MAX_LEN = 128
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
# load github dataframe

with open(train, 'r', encoding='utf-8') as f:
    first_line = f.readline()
if ';' in first_line:
    train_df = pd.read_csv(train, delimiter=";")
    test_df = pd.read_csv(test, delimiter=";")
elif ',' in first_line:
    train_df = pd.read_csv(train, delimiter=",")
    test_df = pd.read_csv(test, delimiter=",")

# train_df = pd.read_csv(train, delimiter=";")
# test_df = pd.read_csv(test, delimiter=";")

print(train_df.head())
print(test_df.head())

tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)

sentences = train_df.Text.values
labels = train_df.Polarity.values

# %%

input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
        str(sent),
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

train_inputs = torch.cat(input_ids, dim=0)
train_masks = torch.cat(attention_masks, dim=0)

label = list(set(labels))
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(label)
labels = label_encoder.transform(labels)
train_labels = torch.tensor(labels, dtype=torch.long)

print('Training data {} {} {}'.format(train_inputs.shape, train_masks.shape, train_labels.shape))

# %%

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Train Model using Google Colab GPU
model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
if torch.cuda.is_available():
    model.cuda()
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

# %%

### Training

begin = time.time()
train_loss_set = []

for _ in trange(EPOCHS, desc="Epoch"):

    model.train()

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_step = len(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(b_input_ids, token_type_ids=None, \
                        attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        train_loss_set.append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        if nb_tr_steps % 5 == 0:
            print("Epoch: {}, Step/Total Step: {}/{},Train loss: {}".format(_, step, total_step, tr_loss / nb_tr_steps))

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

end = time.time()
print('Used {} second'.format(end - begin))

# %%

# Save it to gdrive
os.makedirs(output, exist_ok=True)
print(output)
save_dict = {
    'model':model.state_dict(),
    'label_encoder':label_encoder
}
torch.save(save_dict, f'{output}/RoBERTa.pth')

# %%

# Load pretrained model for specific dataset
model.load_state_dict(torch.load(f'{output}/RoBERTa.pth').get('model'))

# %%

### Test
begin = time.time()
sentences = test_df.Text.values
labels = test_df.Polarity.values
# ids = test_df.id.values

input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
        str(sent),
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

prediction_inputs = torch.cat(input_ids, dim=0)
prediction_masks = torch.cat(attention_masks, dim=0)
labels = label_encoder.transform(labels)
prediction_labels = torch.tensor(labels, dtype=torch.long)

prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

model.eval()
predictions, true_labels = [], []

for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    # print('train', logits)
    label_ids = b_labels.to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

end = time.time()
print('Prediction used {:.2f} seconds'.format(end - begin))

# %%

# Final Results

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]


print("Accuracy of {} on {} is: {}".format(m_name, dataset, metrics.accuracy_score(flat_true_labels, flat_predictions)))

def covert(flat_true_labels):
    zeros = np.zeros(shape=(len(flat_true_labels),len(label_encoder.classes_)))
    for i, v  in enumerate(flat_true_labels):
        zeros[i][v] = 1
    return zeros

print(metrics.classification_report(covert(flat_true_labels), covert(flat_predictions), target_names=label_encoder.classes_))
with open(f'{output}/RoBERTa.txt', "w") as filer:
    filer.write(metrics.classification_report(covert(flat_true_labels), covert(flat_predictions), target_names=label_encoder.classes_))

with open(f'{output}/acc.txt', "w") as filer:
    filer.write(str(metrics.accuracy_score(flat_true_labels, flat_predictions)))

