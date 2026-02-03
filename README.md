# LLM 수리추론 능력 강화를 위한 SFT 학습

LLM의 수학적 추론 능력을 향상시키기 위한 SFT(Supervised Fine-Tuning) 실험 프로젝트입니다.

## 프로젝트 개요

- **목표**: SFT를 활용한 LLM 수리추론 능력 강화
- **학습 데이터**: grade-school-math-instructions, MathQA
- **평가 데이터**: MathQA (lm-evaluation-harness)
- **학습 모델**: Qwen2.5-0.5B, Qwen2.5-1.5B (base models)
- **환경**: Google Colab T4 GPU

## 노트북 구성

### 1. [01_sft_training.ipynb](01_sft_training.ipynb) - 기본 SFT 학습

기본적인 SFT 학습을 수행합니다.

- **데이터셋**: `qwedsacf/grade-school-math-instructions` (8,792 샘플)
- **학습 방식**: QLoRA (4-bit quantization + LoRA)
- **프롬프트 형식**: `### Question: ... ### Answer: ...`
- **최적화**:
  - Gradient Checkpointing
  - SDPA (Scaled Dot Product Attention)
  - Sequence Packing (2~3x 속도 향상)
  - Mixed Precision (fp16)

### 2. [02_sft_training_format_match.ipynb](02_sft_training_format_match.ipynb) - 학습-평가 Objective 일치

학습과 평가의 objective를 일치시켜 성능을 개선합니다.

- **핵심 아이디어**: MathQA 평가는 multiple choice 형식이므로, 학습도 동일한 방식으로 수행
- **데이터 변환**:
  - grade-school-math에서 1,500개 샘플 사용
  - 각 문제당 5개 옵션 생성 (정답 1개 + 오답 4개)
- **학습 방식**:
  - 각 옵션 continuation의 log-likelihood 계산
  - Softmax over options → Cross-entropy loss

### 3. [03_sft_training_mathqa_training.ipynb](03_sft_training_mathqa_training.ipynb) - MathQA 직접 학습

MathQA 데이터셋을 직접 활용한 학습 실험입니다.

- **Scenario 1**: MathQA train만 사용 (1,500개)
- **Scenario 2**: MathQA + Grade-school-math 혼합 (1,500 + 1,500 = 3,000개)
- **평가**: lm-evaluation-harness의 mathqa 태스크 (test split)

### 4. [04_evaluation.ipynb](04_evaluation.ipynb) - 모델 평가

학습된 모델들을 MathQA 태스크로 평가합니다.

**평가 대상 모델:**
| 모델 타입 | 모델명 |
|-----------|--------|
| Base 모델 | Qwen2.5-0.5B, Qwen2.5-1.5B |
| SFT 모델 | Qwen2.5-0.5B-math-SFT, Qwen2.5-1.5B-math-SFT |
| SFT Improved | Qwen2.5-0.5B-math-SFT-Improved (MC objective) |
| Instruct 모델 | Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct |

## 평가 결과

| Model | Accuracy | Std Error |
|-------|----------|-----------|
| Qwen2.5-0.5B (Base) | 0.2874 | 0.0083 |
| Qwen2.5-0.5B-math-SFT | 0.2884 | 0.0083 |
| Qwen2.5-0.5B-Instruct | 0.2901 | 0.0083 |
| Qwen2.5-1.5B (Base) | 0.3461 | 0.0087 |
| Qwen2.5-1.5B-math-SFT | 0.2978 | 0.0084 |
| Qwen2.5-1.5B-Instruct | 0.3374 | 0.0087 |

## 환경 설정

### 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| torch | 2.4.0+ | 딥러닝 프레임워크 |
| transformers | 4.44.2+ | 모델 로딩 |
| datasets | 2.21.0 | 데이터 로딩 |
| accelerate | 0.33.0+ | 분산 학습 |
| peft | 0.12.0 | LoRA/QLoRA |
| bitsandbytes | 0.43.3+ | 4-bit quantization |
| trl | 0.9.6 | SFT Trainer |
| lm-eval | 0.4.3 | 모델 평가 |
| wandb | 0.17.5+ | 실험 추적 |

### 설치

```bash
pip install transformers>=4.45.0 bitsandbytes>=0.44.0
pip install datasets==2.21.0 peft==0.12.0 trl==0.9.6
pip install accelerate>=1.7.0 scipy==1.13.1
pip install lm-eval==0.4.3 wandb
```

### 환경 변수 설정

`.env.example`을 복사하여 `.env` 파일을 생성하고 필요한 값을 입력하세요:

```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_username
```

## 사용 방법

### 1. SFT 학습 실행

```bash
# Google Colab에서 노트북 실행
# 01_sft_training.ipynb → 기본 SFT 학습
# 02_sft_training_format_match.ipynb → MC objective 학습
# 03_sft_training_mathqa_training.ipynb → MathQA 직접 학습
```

### 2. 모델 평가

```bash
# 04_evaluation.ipynb 실행
# lm-evaluation-harness를 사용하여 MathQA 태스크로 평가
```

## 프로젝트 구조

```
snu-dbsa/
├── 01_sft_training.ipynb              # 기본 SFT 학습
├── 02_sft_training_format_match.ipynb # 학습-평가 objective 일치
├── 03_sft_training_mathqa_training.ipynb # MathQA 직접 학습
├── 04_evaluation.ipynb                # 모델 평가
├── .env.example                       # 환경 변수 예시
└── README.md                          # 프로젝트 설명
```

## 핵심 기술

### QLoRA (Quantized Low-Rank Adaptation)
- 4-bit quantization으로 메모리 사용량 감소
- LoRA adapter로 효율적인 fine-tuning
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Multiple Choice Training Objective
- 기존 SFT: 생성(decoding) 방식으로 학습
- 개선된 방식: multiple choice에서 정답 옵션의 log-likelihood가 최대가 되도록 학습
- 평가 방식과 일치하여 더 나은 성능 기대

### 속도 최적화
- SDPA (Scaled Dot Product Attention): T4에서 ~1.5x 빠른 attention
- Sequence Packing: 여러 샘플을 한 시퀀스에 패킹 → 2~3x 속도 향상
- Gradient Checkpointing: 메모리 사용량 감소

## 참고 자료

- [Qwen2.5 모델](https://huggingface.co/Qwen)
- [grade-school-math-instructions 데이터셋](https://huggingface.co/datasets/qwedsacf/grade-school-math-instructions)
- [MathQA 데이터셋](https://huggingface.co/datasets/allenai/math_qa)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
