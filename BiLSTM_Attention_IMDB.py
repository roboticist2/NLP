from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
import os

import tensorflow as tf

vocab_size = 10000 # 최대단어갯수
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

# IMDB 데이터는 이미 정수 인코딩 돼있으므로 패딩만 하면 됨

#print('리뷰의 최대 길이 : {}'.format(max(len(l) for l in X_train)))
#print('리뷰의 평균 길이 : {}'.format(sum(map(len, X_train))/len(X_train)))

max_len = 400 # 패딩길이
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, values, query): # 단, key와 value는 같음
    # query shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


#######모델 설계########

#입력층, 임베딩층 설계
sequence_input = Input(shape=(max_len,), dtype='int32')
embedding_dim = 64
embedded_sequences = Embedding(vocab_size, embedding_dim, input_length=max_len, mask_zero = True)(sequence_input)

# 양방향 LSTM 이므로 두 층 설계
# LSTM 첫번째 층. 두번째 층을 위에 쌓아야 하므로 return_sequences는 True.
lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences = True))(embedded_sequences)

# LSTM 두번째 층. 상태 리턴 받아야 하므로 return_state는 True
lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)

# 순반향 LSTM 은닉 상태, 셀 상태 forward_h, forward_c 저장
# 역방향 LSTM 은닉 상태, 셀 상태 backward_h, backward_c 저장
state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
state_c = Concatenate()([forward_c, backward_c]) # 셀 상태

# 컨텍스트 벡터 계산
attention = BahdanauAttention(64) # 가중치 크기 정의
context_vector, attention_weights = attention(lstm, state_h)

# 컨텍스트 벡터 dense layer 통과
dense1 = Dense(20, activation="relu")(context_vector)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation="sigmoid")(dropout) # 2진 분류기
model = Model(inputs=sequence_input, outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 2진 분류 시그모이드 -> binary_crossentropy

history = model.fit(X_train, y_train, epochs = 3, batch_size = 128, validation_data=(X_test, y_test), verbose=1)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))