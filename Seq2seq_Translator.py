import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np

pd.set_option("io.excel.xlsx.writer", "xlsxwriter")

#http = urllib3.PoolManager()
#url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)

#웹 주소에서 파일 다운
#with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:
#    shutil.copyfileobj(r, out_file)

#다운 받은 zip파일 압축해제
with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

#src : source(from 언어)
#tar : target(to 언어)
lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic'] # lic???

print('전체 샘플의 개수 :',len(lines))
print(type(lines))

num_samples = 30000

lines = lines.loc[:, 'src':'tar']
lines = lines[0:num_samples] # 6만개만 저장 #문장 샘플은 23.04.21 기준 약 22만개
#print(lines.sample(10))

#<sos>와 <eos>역할은 \t와 \n으로 치환
lines.tar = lines.tar.apply(lambda x : '\t '+ x + ' \n')
#print(lines.sample(10))

# 문자 집합 구축
# 토큰 단위가 단어가 아닌 문자임
src_vocab = set()
for line in lines.src: # 1줄씩 읽음
    for char in line: # 1개의 문자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
#print('source 문장의 char 집합 :',src_vocab_size) # 영어의 문자
#print('target 문장의 char 집합 :',tar_vocab_size) # 프랑스어의 문자수

# 정렬
src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
#print(src_vocab[60:79])
#print(tar_vocab[83:102])

# 각 문자에 인덱스 매칭(인코딩 인덱스)
src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
#print(src_to_index)
#print(tar_to_index)

encoder_input = []

# from 언어 인코딩 입력 정수 인코딩
# 1개의 문장
for line in lines.src:
  encoded_line = []
  # 각 줄에서 1개의 char
  for char in line:
    # 각 char을 정수로 변환
    encoded_line.append(src_to_index[char])
  encoder_input.append(encoded_line)
# print('source 문장 :',lines.src[:5]) # ex 1.Go. 2.Go 3. HI.
# print('source 문장의 정수 인코딩 :',encoder_input[:5]) # ex [[30.64.10],[30,64,10],[31,58,10]]

# 디코더의 입력인 to 언어 정수 인코딩
decoder_input = []
for line in lines.tar:
  encoded_line = []
  for char in line:
    encoded_line.append(tar_to_index[char])
  decoder_input.append(encoded_line)
# print('target 문장의 정수 인코딩 :',decoder_input[:5])

# 디코더 예측값의 비교 데이터인 실제값 정수 인코딩(디코더 출력의 정답)
# 실제값에는 문장 맨앞의 <sos>가 빠지고 맨 끝의 <eos>만 남아야 함.
decoder_target = []
for line in lines.tar:
  timestep = 0
  encoded_line = []
  for char in line:
    if timestep > 0:
      encoded_line.append(tar_to_index[char])
    timestep = timestep + 1 # 맨 첫번째는 모두 <sos>인 \t이므로(정수라벨 1번) 첫번째 단어는 건너뛰고 출력하도록
  decoder_target.append(encoded_line)
#print('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])

#데이터간 패딩을 위한 최대문장길이 확인
max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
#print('source 문장의 최대 길이 :',max_src_len)
#print('target 문장의 최대 길이 :',max_tar_len)
#from언어와 to언어끼리 모두 맞출 필요 없이, 각각의 최대길이만큼 패딩을 맞추면 된다.

#데이터 패딩
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

# 원핫 인코딩
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)
# 문자 단위 번역기이므로, 워드 임베딩 사용 안함
# 예측값과의 오차 측정에 사용되는 실제값 뿐만 아닌 입력값도 원핫 벡터 사용
# ???????????????????????????????????????????????????????????

encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True)

# encoder_outputs은 여기서는 불필요
# decoder에는 encoder의 state(h,c)가 전달된다.
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# LSTM은 바닐라 RNN과는 달리 상태가 두 개. 은닉 상태와 셀 상태.
# 참고 : 일반 RNN은 은닉상태만 존재
encoder_states = [state_h, state_c]
#encoder_states를 decoder에 전달하면 h,c모두 전달된다.
#이것이 context vector가 된다.

decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

# 디코더에게 인코더의 은닉 상태, 셀 상태를 전달.
decoder_outputs, _, _= decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=100, epochs=10, validation_split=0.2)

# 디코더도 은닉 상태, 셀 상태를 리턴하기는 하지만 훈련 과정에서는 사용하지 않는다.
# 그 후 출력층에 to언어의 단어 집합의 크기만큼 뉴런을 배치한 후 소프트맥스 함수를 사용하여 실제값과의 오차를 구합니다.
# -> 단어 집합이 몇만개 되는거 아닌가?????????????????

# 인코더 모델 정의
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용.
# 뒤의 함수 decode_sequence()에 동작을 구현 예정
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태(decoder_states)를 버리지 않음.
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

# 인덱스에서 단어 추출
index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())

#index_list = list(range(len(src_to_index)))
#src_to_index = pd.DataFrame.from_dict(src_to_index, orient='index', columns=['Index'])

#index_list = list(range(len(tar_to_index)))
#tar_to_index = pd.DataFrame.from_dict(tar_to_index, orient='index', columns=['Index'])

#src_to_index.to_excel('src_to_index.xlsx', index=False)
#tar_to_index.to_excel('tar_to_index.xlsx', index=False)

def decode_sequence(input_seq):
  # 입력으로부터 인코더의 상태를 얻음
  states_value = encoder_model.predict(input_seq)

  # <SOS>에 해당하는 원-핫 벡터 생성
  target_seq = np.zeros((1, 1, tar_vocab_size))
  target_seq[0, 0, tar_to_index['\t']] = 1.

  stop_condition = False
  decoded_sentence = "" # 예측 문장. 초기엔 아무 문자 없는 상태로 시작

  # stop_condition이 True가 될 때까지 루프 반복
  while not stop_condition:
    # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

    # 예측 결과를 문자로 변환
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_char = index_to_tar[sampled_token_index]

    # 현재 시점의 예측 문자를 예측 문장에 추가
    decoded_sentence += sampled_char

    # <eos>에 도달하거나 최대 길이를 넘으면 중단.
    if (sampled_char == '\n' or
        len(decoded_sentence) > max_tar_len):
        stop_condition = True

    # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, sampled_token_index] = 1.

    # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
    states_value = [h, c]

  return decoded_sentence

for seq_index in [3,50,100,300,1001]: # 입력 문장의 인덱스
  input_seq = encoder_input[seq_index:seq_index+1]
  decoded_sentence = decode_sequence(input_seq)
  print(35 * "-")
  print('입력 문장:', lines.src[seq_index])
  print('정답 문장:', lines.tar[seq_index][2:len(lines.tar[seq_index])-1]) # '\t'와 '\n'을 빼고 출력
  print('번역 문장:', decoded_sentence[1:len(decoded_sentence)-1]) # '\n'을 빼고 출력