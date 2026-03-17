# StudyLens

AI-powered lecture review platform that consolidates slides, transcripts, and notes into structured summaries with RAG-based Q&A.

**Built for AIPI 540 NLP Module Project, Duke University**

- **Live App**: https://studylens.up.railway.app/
- **Repository**: https://github.com/SophiaYifei/StudyLens.git
- **Team**: Yifei Guo ([SophiaYifei](https://github.com/SophiaYifei)), Sharmil Nanjappa ([SharmilNK](https://github.com/SharmilNK))

---

## Project Overview

StudyLens takes three types of input per lecture — slide decks (PPTX), auto-generated transcripts, and student notes — and produces structured summaries for exam review. We evaluate 9 model configurations across three paradigms (naive baselines, classical ML, deep learning) on 10 lecture topics from two Duke graduate courses (AIPI 540: Deep Learning, AIPI 520: Machine Learning).

The deployed web application features two modes:
- **Review Mode**: Real-time AI summarization via DeepSeek API, pre-computed fine-tuned Qwen summaries for comparison, and a RAG-based Q&A chatbot grounded in lecture materials
- **Lecture Mode**: PDF slide viewer with integrated note-taking

---

## Repository Structure

```
StudyLens/
├── README.md                        <- This file
├── requirements.txt                 <- Base environment (BART, Long-T5, evaluation)
├── requirements-llm.txt             <- LLM environment (Qwen2.5-7B)
├── requirements-app.txt             <- Web app dependencies
├── .env.example                     <- API key template (copy to .env)
├── .gitignore
├── LICENSE
├── Makefile
├── main.py                          <- Data processing pipeline entry point
├── app.py                           <- FastAPI web application backend
├── Procfile                         <- Railway deployment config
├── railway.toml                     <- Railway build settings
├── static/
│   └── index.html                   <- Frontend (single-file HTML/CSS/JS)
├── scripts/
│   ├── model.py                     <- All summarization models (naive, TF-IDF, neural, LLM)
│   ├── make_dataset.py              <- Data loading and transcript denoising
│   ├── build_features.py            <- Chunking, embedding, topic assignment
│   ├── naive.py                     <- First-5-slides heuristic baseline
│   ├── eval.py                      <- Unified evaluation (ROUGE-L, BERTScore, NLI)
│   ├── finetune_qwen.py             <- QLoRA fine-tuning pipeline
│   └── plot_from_csv.py             <- Generate all report figures from eval CSV
├── models/
│   └── qwen7b-lora/                 <- Fine-tuned LoRA adapter weights (~40MB)
├── data/                            <- (gitignored, downloaded from HuggingFace at runtime)
│   ├── raw/                         <- Slides, transcripts, notes, files are removed due to privacy
│   ├── processed/                   <- Combined _ori.txt files, files are removed due to privacy
│   ├── reference/                   <- GPT-5.2-generated reference summaries
│   ├── outputs/                     <- Model-generated summaries
│   └── eval/                        <- Evaluation results and plots
└── notebooks/                       <- Exploration notebooks (not graded)
```

---

## Models

All models are implemented in `scripts/model.py` (sharing a common `BaseSummarizer` interface) except the first-5-slides heuristic which is in `scripts/naive.py`.

| Requirement    | Class / File                    | `--model` flag | Description                                    |
|----------------|---------------------------------|----------------|------------------------------------------------|
| Naive baseline | `FirstSentenceSummarizer`       | `first5`       | First sentence from first 5 slides             |
| Naive baseline | `RandomExtractiveSummarizer`    | `random`       | Randomly select 15 sentences                   |
| Classical ML   | `TFIDFExtractiveSummarizer`     | `tfidf`        | TF-IDF sentence ranking, top 15                |
| Deep learning  | `BARTSummarizer`                | `bart`         | BART-CNN, 1024 tokens, news-trained            |
| Deep learning  | `LongT5Summarizer`              | `longt5`       | Long-T5, 16384 tokens, book-trained            |
| Deep learning  | `BARTSamsumSummarizer`          | `bart-samsum`  | BART-SAMSum, 1024 tokens, dialogue-trained     |
| Deep learning  | `LEDArxivSummarizer`            | `led-arxiv`    | LED-arXiv, 16384 tokens, academic paper-trained|
| Deep learning  | `QwenSummarizer`                | `qwen7b`       | Qwen2.5-7B, 128K context, 4-bit quantized     |
| Deep learning  | `finetune_qwen.py`              | via script     | QLoRA fine-tuned Qwen on lecture data           |

---

## Environment Setup

This project uses two environments due to dependency conflicts between encoder-decoder models and Qwen.

### Base environment (BART, Long-T5, BART-SAMSum, LED-arXiv, Evaluation)
```bash
pip install -r requirements.txt
```

### LLM environment (Qwen2.5-7B, requires NVIDIA GPU with 16GB+ VRAM)
```bash
pip install -r requirements-llm.txt
```

> **Why two environments?** BART/Long-T5 require `transformers==4.41.2`, while Qwen2.5-7B requires `transformers>=4.45.0`. Running both in one environment causes version conflicts.

---

## Running the Pipeline

### Step 1: Process raw data
```bash
python main.py
```
This loads raw files from `data/raw/`, denoises transcripts, and produces 10 combined `_ori.txt` files in `data/processed/`.

### Step 2: Generate summaries
```bash
# Base environment:
python scripts/model.py --model first5
python scripts/model.py --model random
python scripts/model.py --model tfidf
python scripts/model.py --model bart --strategy both
python scripts/model.py --model longt5 --strategy both
python scripts/model.py --model bart-samsum --strategy both
python scripts/model.py --model led-arxiv --strategy both

# LLM environment (requires GPU):
python scripts/model.py --model qwen7b
python scripts/model.py --model qwen7b --output_ratio 0.01
python scripts/model.py --model qwen7b --output_ratio 0.03
```

### Step 3: Fine-tune Qwen (LLM environment, A100 GPU)
```bash
python scripts/finetune_qwen.py --mode prepare
python scripts/finetune_qwen.py --mode train --epochs 5
python scripts/finetune_qwen.py --mode inference
```

### Step 4: Evaluate all models
```bash
python scripts/eval.py              # ROUGE-L + BERTScore
python scripts/eval.py --run-nli    # also compute NLI (slower)
```
Results saved to `data/eval/evaluation_results_all.csv` and `data/eval/evaluation_averages.csv`.

### Step 5: Generate plots
```bash
python scripts/plot_from_csv.py
```
Figures saved to `data/eval/plots/`.

---

## Running on Google Colab

Due to GPU requirements, we recommend Colab with an A100 GPU. Use two separate notebooks to avoid dependency conflicts.

**Mount Drive (both notebooks):**
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir("/content/drive/{your directory}")
```

**Notebook A** (base environment):
```python
!pip install -r requirements.txt
!python scripts/model.py --model bart --strategy both
!python scripts/model.py --model longt5 --strategy both
!python scripts/model.py --model bart-samsum --strategy both
!python scripts/model.py --model led-arxiv --strategy both
!python scripts/eval.py --run-nli
```

**Notebook B** (LLM environment, separate session):
```python
!pip install -r requirements-llm.txt
!python scripts/model.py --model qwen7b
!python scripts/finetune_qwen.py --mode train --epochs 5
!python scripts/finetune_qwen.py --mode inference
```

---

## Web Application

The deployed app is at https://studylens.up.railway.app/

### Local development
```bash
pip install -r requirements-app.txt
export OPENROUTER_API_KEY=your-key
python app.py
```

### Deployment (Railway)
The app auto-downloads data from a private HuggingFace dataset repo (`YifeiGuo/studylens-data`) on first startup. Required environment variables on Railway:
- `OPENROUTER_API_KEY` — for real-time summarization and chat
- `HF_TOKEN` — for downloading data from private HuggingFace repo

---

## API Keys

Copy the example environment file and add your credentials:
```bash
cp .env.example .env
```

- `ANTHROPIC_API_KEY` — for ClaudeSummarizer (optional; all other models work without it)
- `OPENROUTER_API_KEY` — for the web app's real-time AI features

The `.env` file is in `.gitignore` and will never be committed.

---

## AI Attribution

Code co-authored with Claude (Anthropic, https://claude.ai) for structural design, debugging, and documentation. All files contain attribution headers.
