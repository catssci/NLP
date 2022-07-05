import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams

from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers import Dot
from keras.utils import plot_model
from IPython.display import SVG

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Reference: https://wikidocs.net/69141

dataset = fetch_20newsgroups(shuffle = True, random_state = 1,
                             remove = ('headers', 'footers', 'quotes'))
documents = dataset.data
print("총 샘플 수 : ", len(documents))

news_df = pd.DataFrame({'document': documents})
# 특수 문자 제거
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ", regex = True)
# 길이가 3이하인 단어 제거
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
# 전체 단어에 대한 소문자 변환
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

# print(news_df.head())

# null data check
# print(news_df.isnull().values.any())
# >> False

# empty data check
# empty -> Null
news_df.replace("", float("NaN"), inplace = True, regex = True)
print(news_df.isnull().values.any())

news_df.dropna(inplace = True)
print("총 샘플 수 : ", len(news_df))

# 불용어 제거
# stopwords 를 사용하기 위해 다음과 같은 과저을 거친다.
# >> import nltk
# >> nltk.download()
# nltk downloader에서 download 한다.
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = tokenized_doc.to_list()

drop_train = [index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1]
tokenized_doc = np.delete(tokenized_doc, drop_train, axis = 0)
print("총 샘플 수 : ", len(tokenized_doc))

# 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_doc)

word2idx = tokenizer.word_index
idx2word = {value: key for key, value in word2idx.items()}
encoded = tokenizer.texts_to_sequences(tokenized_doc)

print(encoded[0])
print(encoded[1])

vocab_size = len(word2idx) + 1
print("단어 집합의 크기 : ", vocab_size)

# negative sampling
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:10]]

# 첫번째 샘플인 skip_grams[0] 내 skipgrams로 형성된 데이터셋 확인
# 윈도우 크기 내에서 중심 단어, 주변 단어의 관계를 가지는 경우에는 1의 레이블을 갖도록 하고,
# 그렇지 않은 경우는 0의 레이블을 가지도록 하여 데이터셋을 구성
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(5):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
          idx2word[pairs[i][0]], pairs[i][0],
          idx2word[pairs[i][1]], pairs[i][1],
          labels[i]))

print('전체 샘플 수 :',len(skip_grams))

# 첫번째 뉴스그룹 샘플에 대해서 생긴 pairs와 labels의 개수
print(len(pairs))
print(len(labels))

# 모든 뉴스그룹 샘플에 대해서 수행
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded]

# Skip-Gram with Negative Sampling 구현
embedding_dim = 100

# 중심 단어를 위한 임베딩
w_inputs = Input(shape = (1, ), dtype = 'int32')
word_embedding = Embedding(vocab_size, embedding_dim)(w_inputs)

# 주변 단어를 위한 임베딩
c_inputs = Input(shape = (1, ), dtype = 'int32')
context_embedding = Embedding(vocab_size, embedding_dim)(c_inputs)


dot_product = Dot(axes=2)([word_embedding, context_embedding])
dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
output = Activation('sigmoid')(dot_product)

model = Model(inputs=[w_inputs, c_inputs], outputs=output)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')
plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True, rankdir='TB')

for epoch in range(1, 6):
    loss = 0
    for _, elem in enumerate(skip_grams):
        first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [first_elem, second_elem]
        Y = labels
        loss += model.train_on_batch(X,Y)
    print('Epoch :',epoch, 'Loss :',loss)