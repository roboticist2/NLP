import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Bert-base의 토크나이저-

#result = tokenizer.tokenize('Here is the sentence I want tokenizing for.')
#print(result)

# BERT의 단어 집합을 vocabulary.txt에 저장
with open('vocabulary.txt', 'w', encoding='utf-8') as f:
  for token in tokenizer.vocab.keys():
    f.write(token + '\n')

df = pd.read_fwf('vocabulary.txt', header=None)
print(df)