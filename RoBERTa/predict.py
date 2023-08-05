import os
import sys
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import time
import numpy as np
from sklearn import metrics
MAX_LEN = 128
BATCH_SIZE = 2
# ./temp.csv ./output/{}/{}/{}.pth ./result.txt
test = sys.argv[1]
cur_model = (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base')
model = cur_model[0].from_pretrained(cur_model[2], num_labels=3)
if torch.cuda.is_available():
    model.cuda()
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Load pretrained model for specific dataset
model.load_state_dict(torch.load(sys.argv[2]).get('model'))
label_encoder = torch.load(sys.argv[2]).get('label_encoder')
# %%
tokenizer = cur_model[1].from_pretrained(cur_model[2], do_lower_case=True)
### Test
begin = time.time()
test_df = pd.read_csv(test, delimiter=";")
sentences = test_df.Text.values
# labels = test_df.Polarity.values
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
# labels = label_encoder.transform(labels)
# prediction_labels = torch.tensor(labels)

prediction_data = TensorDataset(prediction_inputs, prediction_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

model.eval()
predictions, true_labels = [], []

for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    # label_ids = b_labels.to('cpu').numpy()
    # print(logits)
    predictions.append(logits)
    # true_labels.append(label_ids)

end = time.time()
print('Prediction used {:.2f} seconds'.format(end - begin))

# %%

# Final Results

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
# flat_true_labels = [item for sublist in true_labels for item in sublist]

with open(sys.argv[3], "w") as filer:
    filer.write(','.join(label_encoder.inverse_transform(flat_predictions)))