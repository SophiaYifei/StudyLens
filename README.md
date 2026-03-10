# StudyLens
AI-powered lecture review platform that consolidates slides, transcripts, and notes into structured summaries with RAG-based Q&amp;A — Built for AIPI540 NLP Module Project

## Environment Setup

This project uses two environments due to dependency conflicts between
summarization models (BART/Long-T5) and LLM models (Qwen2.5-7B).

### Option 1: Base environment (BART, Long-T5, BART-SAMSum, LED-arXiv, Claude, Evaluation)
```bash
pip install -r requirements.txt
```

### Option 2: LLM environment (Qwen2.5-7B, requires NVIDIA GPU with 16GB+ VRAM)
```bash
pip install -r requirements-llm.txt
```

> **Why two environments?** BART/Long-T5 require `transformers==4.41.2` for
> stable inference, while Qwen2.5-7B requires `transformers>=4.45.0` for
> model loading and 4-bit quantization support. Running both in one
> environment causes version conflicts.

---

## Running Models

All models are implemented in `scripts/model.py` and share the same
pipeline interface. Use `--model` to select which model to run and
`--strategy` to control summarization behavior.

### Available models

| Model           | `--model` flag  | Environment | Description                                |
|-----------------|-----------------|-------------|--------------------------------------------|
| BART-CNN        | `bart`          | Base        | News-trained, 1024 token limit, chunking   |
| Long-T5         | `longt5`        | Base        | Book-trained, 16384 token limit            |
| BART-SAMSum     | `bart-samsum`   | Base        | Dialogue-trained, 1024 token limit         |
| LED-arXiv       | `led-arxiv`     | Base        | Academic paper-trained, 16384 token limit  |
| Qwen2.5-7B     | `qwen7b`        | LLM         | Instruction-tuned LLM, 128K context, 4-bit quantized |

### Generate summaries
```bash
# Base environment — run any combination:
python scripts/model.py --model bart --strategy final
python scripts/model.py --model bart --strategy concat
python scripts/model.py --model bart --strategy both        # runs concat + final
python scripts/model.py --model longt5 --strategy both
python scripts/model.py --model bart-samsum --strategy both
python scripts/model.py --model led-arxiv --strategy both

# LLM environment — Qwen always uses "final" (ignored if --strategy is set):
python scripts/model.py --model qwen7b

# Custom input/output directories:
python scripts/model.py --model bart --strategy final --input_dir data/processed --output_dir data/outputs
```

### Run on Google Colab

Due to GPU memory requirements, we recommend running on Colab with an A100 GPU.
Use two separate notebooks to avoid dependency conflicts.

**Step 1: Mount Google Drive and navigate to project directory (both notebooks):**
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir("/content/drive/{your directory}")
```

**Step 2a: Notebook A** (base environment — BART, Long-T5, BART-SAMSum, LED-arXiv, Evaluation):
```python
!pip install -r requirements.txt
!python scripts/model.py --model bart --strategy both
!python scripts/model.py --model longt5 --strategy both
!python scripts/model.py --model bart-samsum --strategy both
!python scripts/model.py --model led-arxiv --strategy both
!python scripts/eval.py
```

**Step 2b: Notebook B** (LLM environment — Qwen2.5-7B, requires A100 GPU):
```python
!pip install -r requirements-llm.txt
!python scripts/model.py --model qwen7b
```

> **Note:** Notebooks A and B must be run in separate Colab sessions
> because they require different versions of the `transformers` library.
> Do not run both in the same session.

### Run evaluation
```bash
# Base environment only:
python scripts/eval.py
```

Evaluation auto-discovers all `*_sum_*.txt` files in `data/outputs/`
and computes ROUGE-L, NLI factual consistency, and BERTScore for each.
Results are saved to `data/outputs/evaluation_results.csv`.

### Configure API keys

This project uses the Anthropic API for the Claude-based summarizer.
API keys are **never committed to the repository**.

Copy the example environment file and fill in your credentials:

cp .env.example .env

Then open `.env` and replace the placeholder with your actual key:

ANTHROPIC_API_KEY=your-key-here

You can obtain an API key at https://console.anthropic.com.

> **Note:** The `.env` file is listed in `.gitignore` and will never be
> tracked by Git. Do not remove this entry.

If you do not have an API key, all models except `ClaudeSummarizer` will
still run without any configuration.
