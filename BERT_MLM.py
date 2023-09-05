from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer

# 모델과 tokenizer는 항상 맵핑 관계
# 맵핑되는 정수 인코딩이 다르면 안됨

# [MASK]라고 되어있는 단어를 맞추기 위한 MLM 모델 구조로 BERT를 로드
model = TFBertForMaskedLM.from_pretrained('bert-large-uncased')
# 해당 모델이 학습되었을 당시에 사용된 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

inputs = tokenizer('Soccer is [MASK] really fun [MASK].', return_tensors='tf')
print(inputs['input_ids'])