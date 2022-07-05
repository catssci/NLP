import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt

# NPLM 튜토리얼
# 목적: NPLM의 구조를 이해하자!

# Reference: https://kthworks.github.io/nlp/Neural-Probabilistic-Language-Model-(NPLM)-Pytorch%EB%A1%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0/
# # -- Tokenization 연습 --
# # n-gram (n = 3)
# corpus = ['코딩은 시작이 반이다', '나는 오늘도 주짓수', '코딩은 어려워 짜증나', 'NLP 니가 뭔데', '내가 바로 공주다']
#
# tokens = " ".join(corpus).split()
# tokens = list(set(tokens))
#
# # word -> index / index -> word
# word_dict = {w: i for i, w in enumerate(tokens)}
# index_dict = {i: w for i, w in enumerate(tokens)}

# input, target word 나누기
def make_input_target():
    input_batch = []
    target_batch = []

    for sentence in corpus:
        word = sentence.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)

    return torch.LongTensor(input_batch), torch.LongTensor(target_batch)

# make_input_target check
# input_batch, target_batch = make_input_target()
# print("Input: ", input_batch)
# print("Target: ", target_batch)

# NPLM model
class NPLM(nn.Module):
    def __init__(self):
        super(NPLM, self).__init__()
        self.C = nn.Embedding(V, m)
        self.H = nn.Linear((n-1)*m, n_hidden, bias = False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, V, bias = False)
        self.W = nn.Linear((n-1)*m, V, bias = False)
        self.b = nn.Parameter(torch.ones(V))

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, (n - 1) * m)
        tanh = torch.tanh(self.d + self.H(X))
        output = self.b + self.W(X) + self.U(tanh)
        return output

if __name__ == '__main__':
    corpus = ['코딩은 시작이 반이다', '나는 오늘도 주짓수', '코딩은 어려워 짜증나', 'NLP 니가 뭔데', '내가 바로 공주다']

    tokens = " ".join(corpus).split()
    tokens = list(set(tokens))

    word_dict = {w: i for i, w in enumerate(tokens)}
    index_dict = {i: w for i, w in enumerate(tokens)}

    n = 3
    n_hidden = 2
    m = 2
    V = len(tokens)

    model = NPLM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    input_batch, target_batch = make_input_target()

    # Training
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))  # 1000번째 epoch

        loss.backward()
        optimizer.step()


# Prediction
predict = model(input_batch).data.max(1, keepdim=True)[1]

for i in corpus:
    print(i.split()[:n-1], ' -> ', i.split()[-1])

# Scatter embedded vectors
plt.rc('font', family='Malgun Gothic') # 한글 출력을 가능하게 만들기
plt.rc('axes', unicode_minus=False)   # 한글 출력을 가능하게 만들기

fig, ax = plt.subplots()
ax.scatter(model.C.weight[:,0].tolist(), model.C.weight[:,1].tolist())

for i, txt in enumerate(tokens):
    ax.annotate(txt, (model.C.weight[i,0].tolist(), model.C.weight[i,1].tolist()))

plt.show()