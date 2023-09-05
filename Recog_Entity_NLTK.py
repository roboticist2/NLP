import nltk
nltk.download('words')
from nltk import word_tokenize, pos_tag, ne_chunk

sentence = "James Bond is pure english man and he is working for MI6 as a secret agent"
# 토큰화 후 품사 태깅
tokenized_sentence = pos_tag(word_tokenize(sentence))
print(tokenized_sentence)

ner_sentence = ne_chunk(tokenized_sentence)
print(ner_sentence)