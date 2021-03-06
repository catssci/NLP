from transformers import BertTokenizer
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
# import urllib.request

# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

print(train_data.shape)
print(train_data.groupby('label').count())

# bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)

data = train_data['document'].values
documents = ["[CLS] " + str(d) + " [SEP]" for d in data]
tokenized_doc = [tokenizer.tokenize(s) for s in documents]

print(documents[:3])
print(tokenized_doc[:3])

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_doc]
input_id = pad_sequences(input_ids, maxlen = 128, dtype = 'long',
                         truncating = 'post', padding = 'post')

print(len(input_ids))
print(input_ids[:3])

print(input_id.shape)
print(input_id[:3])

attention_masks = []
for seq in input_id:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)
print(attention_masks[:3])

data = TensorDataset(torch.tensor(input_id), torch.tensor(attention_masks), torch.tensor(train_data['label'].values))
data_seq = TensorDataset(torch.tensor(input_id), torch.tensor(attention_masks), torch.tensor([-1] * len(train_data)))

# RandomSampler: shuffle, SequentialSampler: No-Shuffle
sampler = RandomSampler(data)
seq_sampler = SequentialSampler(data_seq)
data_loader = DataLoader(data, sampler = sampler, batch_size = 3)
data_loader_seq = DataLoader(data_seq, sampler = seq_sampler, batch_size = 3)

for i, b in enumerate(data_loader):
    if i == 1:
        break
    x = b[0]
    y = b[1]
    z = b[2]
    print(x)
    print(y)
    print(z)

for i, b in enumerate(data_loader_seq):
    if i == 1:
        break
    x = b[0]
    y = b[1]
    z = b[2]
    print(x)
    print(y)
    print(z)