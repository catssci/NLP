from transformers import BertTokenizer
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
# import urllib.request

# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

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