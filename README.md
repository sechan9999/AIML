# 🤖 AI/ML Paper Implementations

AI/ML 핵심 개념과 논문 구현을 위한 학습 저장소입니다.

## 📚 목차

1. [BERT & Attention 데모](#bert--attention-데모)

---

## BERT & Attention 데모

### 📄 파일: `bert_attention_demo.py`

이 스크립트는 두 가지 핵심 NLP 개념을 시연합니다:

### 1️⃣ BERT Tokenization (Pre-training)

BERT는 문장을 토큰으로 분리하고 각 토큰에 고유 ID를 부여합니다.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Machine learning is powerful."
inputs = tokenizer(text, return_tensors="pt")
```

**출력 예시:**
```
입력 문장: Machine learning is powerful.
토큰 ID: tensor([[ 101, 3698, 4083, 2003, 3928, 1012, 102]])
```

| 토큰 ID | 의미 |
|---------|------|
| 101 | [CLS] - 문장 시작 |
| 3698 | "machine" |
| 4083 | "learning" |
| 2003 | "is" |
| 3928 | "powerful" |
| 1012 | "." |
| 102 | [SEP] - 문장 끝 |

### 2️⃣ Simple Attention (Attention Is All You Need)

"Attention Is All You Need" 논문의 핵심 메커니즘을 단순화한 구현입니다.

```python
def simple_attention(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights
```

**Attention 수식:**
```
Attention(Q, K, V) = softmax(Q × K^T) × V
```

**출력 예시 (3개 단어 간 집중도):**
```
[[[0.9999  0.0000  0.0001]
  [0.0000  1.0000  0.0000]
  [0.0000  0.0000  1.0000]]]
```

각 단어가 다른 단어에 얼마나 "집중(attention)"하는지를 나타내는 가중치 행렬입니다.

---

## 🛠️ 설치 및 실행

### 필수 패키지 설치
```bash
pip install torch transformers
```

### 실행
```bash
python bert_attention_demo.py
```

---

## 📖 참고 논문

1. **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)
2. **Transformer**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

---

## 📅 업데이트 로그

- **2026-01-19**: BERT Tokenization & Simple Attention 데모 추가

---

## 📜 License

MIT License
