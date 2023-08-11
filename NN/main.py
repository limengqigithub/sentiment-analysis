import sys
import torch.nn as nn
import os
import time
import numpy as np
from tqdm import trange
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from models import NN
from torch.optim import AdamW
import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn import preprocessing
from sklearn import metrics
# stop_words = []
train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]
dataset_name = sys.argv[4]
# train = '../data/two/train.csv'
# test = '../data/two/test.csv'
# output = '../output/two/NN'
# dataset_name = 'two'

MAX_LEN = 64
BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 3
cur_model = './NN/roberta-base'

with open(train, 'r', encoding='utf-8') as f:
    first_line = f.readline()
if ';' in first_line:
    train_df = pd.read_csv(train, delimiter=";")
    test_df = pd.read_csv(test, delimiter=";")
elif ',' in first_line:
    train_df = pd.read_csv(train, delimiter=",")
    test_df = pd.read_csv(test, delimiter=",")

tokenizer = AutoTokenizer.from_pretrained(cur_model, do_lower_case=True)

sentences = train_df.Text.values
labels = train_df.Polarity.values

input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(str(sent), add_special_tokens=True,
                                         max_length=MAX_LEN,
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         return_tensors='pt')

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

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

model = NN()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

loss_function = torch.nn.CrossEntropyLoss()

train_loss_set = []
for _ in trange(EPOCHS, desc="Epoch"):

    model.train()

    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_step = len(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        # Forward pass
        outputs = model(b_input_ids)
        loss = loss_function(outputs, b_labels)
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


# Save it to gdrive
os.makedirs(output, exist_ok=True)
print(output)
save_dict = {
    'model':model.state_dict(),
    'label_encoder':label_encoder
}
torch.save(save_dict, f'{output}/NN.pth')

# %%

# Load pretrained model for specific dataset
model.load_state_dict(torch.load(f'{output}/NN.pth').get('model'))

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
    batch = tuple(t for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids)
        logits = outputs
    logits = torch.softmax(logits, dim=-1)
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


print("Accuracy is: {}".format(metrics.accuracy_score(np.array(flat_true_labels), flat_predictions)))

print(metrics.classification_report(flat_true_labels, flat_predictions))
with open(f'{output}/NN.txt', "w") as filer:
    filer.write(metrics.classification_report(flat_true_labels, flat_predictions))

with open(f'{output}/acc.txt', "w") as filer:
    filer.write(str(metrics.accuracy_score(flat_true_labels, flat_predictions)))

