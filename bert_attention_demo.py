import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# 1. BERT 프리뷰 (Pre-training)
# 문장을 벡터로 변환하여 문맥을 이해하는 과정입니다.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Machine learning is powerful."
inputs = tokenizer(text, return_tensors="pt")

print(f"--- BERT Tokenization Preview ---")
print(f"입력 문장: {text}")
print(f"토큰 ID: {inputs['input_ids']}\n")

# 2. Simple Attention (Attention Is All You Need)
# 단어 간의 연관성을 계산하는 아주 단순화된 구조입니다.
def simple_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights

# 임의의 벡터 생성 (1개 문장, 3개 단어, 8차원 특징)
q = k = v = torch.randn(1, 3, 8)
output, weights = simple_attention(q, k, v)

print(f"--- Attention Weights Preview ---")
print(f"단어 간 집중도(Weights):\n{weights.detach().numpy()}")
