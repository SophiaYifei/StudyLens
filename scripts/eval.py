"""
eval.py - Evaluation of generated summaries using three metrics.

Metric 1  ROUGE-L     Surface-level overlap vs. reference summaries.
Metric 2  NLI         Factual consistency vs. original slide text.
Metric 3  BERTScore   Semantic similarity vs. reference summaries.

Baseline setup (course project):
- Naive        : first-5-sentences heuristic.
- Classical    : TF-IDF extractive.
- Small models : BART (concat), LED-arxiv (concat).
- LLM          : Qwen-7B at multiple ratios (0.06, 0.09, 0.15, 0.30).

This script discovers all available (source, reference, summary) triples
following the current directory conventions and writes a CSV with metrics
per topic, model, and (where applicable) ratio.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional
from math import pi
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import nltk
import matplotlib.pyplot as plt
nltk.download("punkt_tab", quiet=True)


# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR         = Path(__file__).resolve().parent.parent
SOURCE_DIR       = ROOT_DIR / "data" / "processed"   # slide text (ori)
REFERENCE_DIR    = ROOT_DIR / "data" / "reference"   # reference summaries (ref)
SUMMARY_DIR      = ROOT_DIR / "data" / "outputs"     # model-generated summaries

# All evaluation artefacts (CSVs, plots, RAG eval files) are kept under data/outputs/eval
EVAL_DIR             = SUMMARY_DIR / "eval"
PLOTS_DIR            = EVAL_DIR / "plots"
RAG_EVAL_DATA_PATH   = EVAL_DIR / "rag_eval_data.jsonl"
RAG_EVAL_RESULTS_PATH = EVAL_DIR / "rag_eval_results.csv"

# Batch evaluation: set to run only a subset of topics (faster).
#   ML batch 1:  EVAL_TOPICS = ["ml_s1", "ml_s2", "ml_s3", "ml_s4", "ml_s5"], BATCH_SUFFIX = "ml"
#   DL batch 2:  EVAL_TOPICS = ["dl_s1", "dl_s2", "dl_s3", "dl_s4", "dl_s5"], BATCH_SUFFIX = "dl"
#   All topics : EVAL_TOPICS = None, BATCH_SUFFIX = None
#   Test batch : EVAL_TOPICS = ["dl_s1"], BATCH_SUFFIX = "test"
EVAL_TOPICS: Optional[List[str]] = None        # evaluate all discovered topics
BATCH_SUFFIX: Optional[str] = None            # suffix based on full set

# Optional human-readable topic names (fallback to key if missing)
TOPIC_NAMES = {
    "dl_s1": "Introduction to Deep Learning",
    "dl_s2": "Computer Vision 1",
    "dl_s3": "Computer Vision 2",
    "dl_s4": "NLP 1",
    "dl_s5": "NLP 2",
    "ml_s1": "Machine Learning Intro",
    "ml_s2": "Supervised Learning",
    "ml_s3": "Unsupervised Learning",
    "ml_s4": "Evaluation & Model Selection",
    "ml_s5": "Advanced Topics in ML",
}

# NLI settings
NLI_MODEL_NAME     = "cross-encoder/nli-distilroberta-base"
NLI_PREMISE_TOKENS = 400   # max tokens per source chunk (leaves room for hypothesis in 512 limit)
NLI_TOP_K          = 5     # only check top-k most relevant source chunks per sentence

# Max tokens for BERTScore / long texts (avoid "sequence longer than 512" errors and OOM)
MAX_TOKENS_FOR_SCORE = 450   # safe for 512-token models; truncate by word count (~1 word ≈ 1.3 tokens)

# Metric toggles (set RUN_NLI=False to do a fast ROUGE+BERTScore pass)
RUN_NLI = False

# NLI-only run: load existing ROUGE+BERT CSV, run NLI for NLI_MODELS, merge and save. No ROUGE/BERT recomputation.
NLI_ONLY = False

# When RUN_NLI=True, only run NLI for these models (saves time). Set to None to run for all.
# For this run, we evaluate NLI for all non-Qwen models.
#NLI_MODELS = frozenset({"naive_first5", "tfidf", "bart", "led-arxiv"})

# Models to include in plots (set to None to plot all).
# Current setting: show only baselines (no Qwen) in ROUGE/BERT/NLI bar charts.
#PLOT_MODELS = frozenset({"naive_first5", "tfidf", "bart", "led-arxiv"})

# ------------------------------------------------------------------------
# Preset configs (uncomment one block to switch behaviour quickly):
#
# 1) Full evaluation (ROUGE + BERTScore + NLI for ALL models, plots for ALL models)
# RUN_NLI = True
# NLI_ONLY = False
# NLI_MODELS = None
# PLOT_MODELS = None
#
# 2) Fast scoring only (ROUGE + BERTScore, NO NLI, plots for ALL models)
# RUN_NLI = False
# NLI_ONLY = False
NLI_MODELS = None
PLOT_MODELS = None
#
# 3) Full eval with NLI + plots only for baselines (naive, tfidf, bart, led-arxiv)  ← current setting
# RUN_NLI = True
# NLI_ONLY = False
# NLI_MODELS = frozenset({"naive_first5", "tfidf", "bart", "led-arxiv"})
# PLOT_MODELS = frozenset({"naive_first5", "tfidf", "bart", "led-arxiv"})
#
# 4) NLI-only for Qwen at ratios 0.06, 0.09, 0.15, 0.30 (requires prior full ROUGE/BERT run; plots for Qwen only)
# RUN_NLI = True
# NLI_ONLY = True
# NLI_MODELS = frozenset({"qwen7b"})
# PLOT_MODELS = frozenset({"qwen7b"})
# ------------------------------------------------------------------------


# ── Metric 1: ROUGE-L ───────────────────────────────────────────────────

def evaluate_rouge_l(summary: str, reference: str) -> Dict[str, float]:
    """
    ROUGE-L: longest common subsequence between summary and reference.
    Uses stemming for fairer comparison across morphological variants.
    Returns precision, recall, F1.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {
        "rouge_l_precision": round(scores["rougeL"].precision, 4),
        "rouge_l_recall":    round(scores["rougeL"].recall, 4),
        "rouge_l_f1":        round(scores["rougeL"].fmeasure, 4),
    }


# ── Metric 2: NLI Factual Consistency ───────────────────────────────────

def _build_source_chunks(source: str, tokenizer, max_tokens: int) -> List[str]:
    """Split source into sentence-grouped chunks of at most max_tokens."""
    sentences = sent_tokenize(source)
    chunks, buf, buf_tok = [], [], 0
    for sent in sentences:
        sent_tok = len(tokenizer.encode(sent, add_special_tokens=False))
        if buf_tok + sent_tok > max_tokens and buf:
            chunks.append(" ".join(buf))
            buf, buf_tok = [sent], sent_tok
        else:
            buf.append(sent)
            buf_tok += sent_tok
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _find_top_chunks(hypothesis: str, chunks: List[str], top_k: int) -> List[str]:
    """Return the top_k source chunks most similar to hypothesis (Jaccard word overlap)."""
    hyp_words = set(hypothesis.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        union = hyp_words | chunk_words
        scores.append(len(hyp_words & chunk_words) / len(union) if union else 0.0)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def evaluate_nli(
    summary: str,
    source: str,
    nli_model=None,
    nli_tokenizer=None,
) -> Dict[str, float]:
    """
    NLI-based factual consistency.
    For each summary sentence, find the top-k most relevant source chunks,
    then check entailment using roberta-large-mnli.

    Returns:
      entailment_ratio      fraction of sentences entailed (score > 0.5)
      avg_entailment_score  mean of per-sentence max entailment probabilities
      contradiction_ratio   fraction of sentences contradicted (score > 0.5)
    """
    # Load model if not provided
    if nli_model is None or nli_tokenizer is None:
        nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        nli_model.eval()

    summary_sents = sent_tokenize(summary)
    if not summary_sents:
        return {"entailment_ratio": 0.0, "avg_entailment_score": 0.0, "contradiction_ratio": 0.0}

    source_chunks = _build_source_chunks(source, nli_tokenizer, NLI_PREMISE_TOKENS)

    entailment_scores = []
    contradiction_flags = []

    # Hypothesis chunks must fit with premise in 512 tokens (premise up to NLI_PREMISE_TOKENS).
    # Chunk long summary sentences instead of truncating so we don't lose context.
    hyp_max_words = 100

    for hyp in summary_sents:
        # Chunk long hypotheses (preserve full content like BERTScore chunking).
        hyp_chunks = _chunk_text_by_max_words(hyp, max_words=hyp_max_words)
        if not hyp_chunks:
            hyp_chunks = [_truncate_to_max_words(hyp, max_words=hyp_max_words)]

        sent_entailments = []
        sent_contradicted = False

        for hyp_part in hyp_chunks:
            top_chunks = _find_top_chunks(hyp_part, source_chunks, NLI_TOP_K)
            max_entail = 0.0
            for premise in top_chunks:
                inputs = nli_tokenizer(
                    premise,
                    hyp_part,
                    return_tensors="pt",
                    truncation="longest_first",
                    max_length=512,
                    padding=False,
                )
                with torch.no_grad():
                    logits = nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                max_entail = max(max_entail, probs[1].item())
                if probs[0].item() > 0.5:
                    sent_contradicted = True
            sent_entailments.append(max_entail)

        # One entailment score per summary sentence (mean over its chunks); one contradiction flag.
        entailment_scores.append(float(np.mean(sent_entailments)))
        contradiction_flags.append(sent_contradicted)

    n = len(entailment_scores)
    return {
        "entailment_ratio":     round(sum(1 for s in entailment_scores if s > 0.5) / n, 4),
        "avg_entailment_score": round(float(np.mean(entailment_scores)), 4),
        "contradiction_ratio":  round(sum(contradiction_flags) / n, 4),
    }


# ── Metric 3: BERTScore ─────────────────────────────────────────────────

def _truncate_to_max_words(text: str, max_words: int = MAX_TOKENS_FOR_SCORE) -> str:
    """Truncate text to at most max_words (used for NLI hypothesis cap, not for BERTScore)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _chunk_text_by_max_words(text: str, max_words: int = MAX_TOKENS_FOR_SCORE) -> List[str]:
    """
    Split text into chunks of at most max_words, respecting sentence boundaries.
    Ensures no chunk exceeds the model limit while preserving full content across chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    buf, buf_words = [], 0
    for sent in sentences:
        sent_words = len(sent.split())
        if sent_words > max_words:
            # Single sentence too long: flush current buf, then split this sentence by words
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_words = [], 0
            words = sent.split()
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i : i + max_words])
                chunks.append(chunk)
            continue
        if buf_words + sent_words > max_words and buf:
            chunks.append(" ".join(buf))
            buf, buf_words = [sent], sent_words
        else:
            buf.append(sent)
            buf_words += sent_words
    if buf:
        chunks.append(" ".join(buf))
    return chunks if chunks else [text.strip() or " "]


def evaluate_bertscore(summary: str, reference: str) -> Dict[str, float]:
    """
    BERTScore: token-level semantic similarity using contextual embeddings.
    Uses roberta-large with baseline rescaling. Long reference/summary are split
    into chunks (by sentence, under 512 tokens each); aligned chunk pairs are
    scored and averaged so the full text is compared without losing context.
    """
    ref_chunks = _chunk_text_by_max_words(reference)
    sum_chunks = _chunk_text_by_max_words(summary)
    n = min(len(ref_chunks), len(sum_chunks))
    if n == 0:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

    cands = sum_chunks[:n]
    refs = ref_chunks[:n]
    P, R, F1 = bert_score_fn(
        cands, refs,
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )
    return {
        "bertscore_precision": round(float(P.mean().item()), 4),
        "bertscore_recall":    round(float(R.mean().item()), 4),
        "bertscore_f1":        round(float(F1.mean().item()), 4),
    }


# ── File discovery ───────────────────────────────────────────────────────

def _topic_id_from_source_path(path: Path) -> str:
    """
    Convert e.g. dl_s1_ori.txt -> dl_s1
    """
    stem = path.stem
    if stem.endswith("_ori"):
        return stem[:-4]
    return stem


def discover_evaluation_triples() -> List[Dict]:
    """
    Find all (source, reference, summary) triples for the baseline models.

    Models / layouts assumed:
      - Naive first5:
          data/outputs/naive/first5/{topic_id}_naive.txt
      - Classical TF-IDF:
          data/outputs/classical_ml/tfidf/{topic_id}_sum_classical_ml_tfidf.txt
      - BART concat:
          data/outputs/neural_network/bart/concat/{topic_id}_sum_bart_concat.txt
      - LED-arxiv concat:
          data/outputs/neural_network/led-arxiv/concat/{topic_id}_sum_led-arxiv_concat.txt
      - Qwen7b ratios (0.06, 0.09, 0.15, 0.30):
          data/outputs/neural_network/qwen7b/ratioXX/{topic_id}_sum_qwen7b_ratioXX.txt
          where XX ∈ {06, 09, 15, 30}.
    """
    triples: List[Dict] = []

    # Discover topics from available source files (optionally restricted by EVAL_TOPICS)
    source_paths = sorted(SOURCE_DIR.glob("*_ori.txt"))
    if EVAL_TOPICS:
        source_paths = [p for p in source_paths if _topic_id_from_source_path(p) in EVAL_TOPICS]
        print(f"  Batch mode: evaluating {len(EVAL_TOPICS)} topic(s) only.")

    for source_path in source_paths:
        topic_id = _topic_id_from_source_path(source_path)
        topic_name = TOPIC_NAMES.get(topic_id, topic_id)

        ref_path = REFERENCE_DIR / f"{topic_id}_ref.txt"
        if not ref_path.exists():
            print(f"  WARNING: Reference not found for {topic_id}: {ref_path}")
            continue

        # 1) Naive first5
        naive_path = SUMMARY_DIR / "naive" / "first5" / f"{topic_id}_naive.txt"
        if naive_path.exists():
            triples.append(
                {
                    "topic":         topic_name,
                    "topic_key":     topic_id,
                    "model":         "naive_first5",
                    "strategy":      "naive",
                    "ratio":         None,
                    "source_path":   source_path,
                    "reference_path": ref_path,
                    "summary_path":  naive_path,
                }
            )
        else:
            print(f"  WARNING: Naive first5 summary not found: {naive_path}")

        # 2) Classical TF-IDF
        tfidf_path = (
            SUMMARY_DIR
            / "classical_ml"
            / "tfidf"
            / f"{topic_id}_sum_classical_ml_tfidf.txt"
        )
        if tfidf_path.exists():
            triples.append(
                {
                    "topic":         topic_name,
                    "topic_key":     topic_id,
                    "model":         "tfidf",
                    "strategy":      "tfidf",
                    "ratio":         None,
                    "source_path":   source_path,
                    "reference_path": ref_path,
                    "summary_path":  tfidf_path,
                }
            )
        else:
            print(f"  WARNING: TF-IDF summary not found: {tfidf_path}")

        # 3) BART concat
        bart_concat_path = (
            SUMMARY_DIR
            / "neural_network"
            / "bart"
            / "concat"
            / f"{topic_id}_sum_bart_concat.txt"
        )
        if bart_concat_path.exists():
            triples.append(
                {
                    "topic":         topic_name,
                    "topic_key":     topic_id,
                    "model":         "bart",
                    "strategy":      "concat",
                    "ratio":         None,
                    "source_path":   source_path,
                    "reference_path": ref_path,
                    "summary_path":  bart_concat_path,
                }
            )
        else:
            print(f"  WARNING: BART concat summary not found: {bart_concat_path}")

        # 4) LED-arxiv concat
        led_concat_path = (
            SUMMARY_DIR
            / "neural_network"
            / "led-arxiv"
            / "concat"
            / f"{topic_id}_sum_led-arxiv_concat.txt"
        )
        if led_concat_path.exists():
            triples.append(
                {
                    "topic":         topic_name,
                    "topic_key":     topic_id,
                    "model":         "led-arxiv",
                    "strategy":      "concat",
                    "ratio":         None,
                    "source_path":   source_path,
                    "reference_path": ref_path,
                    "summary_path":  led_concat_path,
                }
            )
        else:
            print(f"  WARNING: LED-arxiv concat summary not found: {led_concat_path}")

        # 5) Qwen7b with selected ratios
        qwen_ratios = ["ratio06", "ratio09", "ratio15", "ratio30"]
        for ratio in qwen_ratios:
            ratio_dir = SUMMARY_DIR / "neural_network" / "qwen7b" / ratio
            qwen_path = ratio_dir / f"{topic_id}_sum_qwen7b_{ratio}.txt"
            if not qwen_path.exists():
                print(f"  WARNING: Qwen summary not found: {qwen_path}")
                continue

            triples.append(
                {
                    "topic":         topic_name,
                    "topic_key":     topic_id,
                    "model":         "qwen7b",
                    "strategy":      "ratio",
                    "ratio":         ratio,
                    "source_path":   source_path,
                    "reference_path": ref_path,
                    "summary_path":  qwen_path,
                }
            )

    return triples


# ── Plotting helpers ──────────────────────────────────────────────────────

def _plot_metric_bars(df: pd.DataFrame, metric: str, ylabel: str, filename: Path) -> None:
    """
    Create a grouped bar plot of a metric by (model, ratio).

    For non-ratio models (naive, tfidf, bart, led-arxiv) the ratio column
    will typically be NaN; these are displayed as a single group.
    """
    if metric not in df.columns:
        print(f"  Metric {metric} not in dataframe; skipping plot.")
        return

    # Optionally restrict to a subset of models for plotting.
    plot_df = df
    if "PLOT_MODELS" in globals() and PLOT_MODELS is not None:
        plot_df = plot_df[plot_df["model"].isin(PLOT_MODELS)]

    # For NLI-style metrics, drop rows where the metric is NaN (e.g. models
    # without NLI scores), otherwise the plot will show flat zero bars.
    if metric in {"entailment_ratio", "avg_entailment_score", "contradiction_ratio"}:
        plot_df = plot_df.dropna(subset=[metric])
        if plot_df.empty:
            print(f"  Metric {metric} has no non-NaN values; skipping plot.")
            return

    agg = (
        plot_df.groupby(["model", "ratio"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    models = agg["model"].unique()
    ratios = [r for r in agg["ratio"].unique() if isinstance(r, str)] or [None]

    x = np.arange(len(models))
    width = 0.15 if len(ratios) > 1 else 0.6

    plt.figure(figsize=(10, 5))
    for idx, ratio in enumerate(ratios):
        sub = agg[agg["ratio"] == ratio] if ratio is not None else agg[agg["ratio"].isna()]
        heights = []
        for m in models:
            row = sub[sub["model"] == m]
            heights.append(row[metric].iloc[0] if not row.empty else 0.0)
        offsets = x + (idx - (len(ratios) - 1) / 2) * width
        label = ratio if ratio is not None else "no-ratio"
        plt.bar(offsets, heights, width=width, label=label)

    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by model and ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Plot saved: {filename}")


# ── RAG evaluation (ragas) ────────────────────────────────────────────────

def _run_rag_evaluation() -> Optional[pd.DataFrame]:
    """
    Run RAG evaluation using ragas if a JSONL dataset is present.

    Expected format of rag_eval_data.jsonl:
      - question: str
      - contexts: list[str]
      - answer: str
      - ground_truth: str
      - model: str (name of the model that produced the answer)

    Returns a dataframe with columns:
      model, faithfulness, answer_relevancy, context_precision, context_recall
    """
    if not RAG_EVAL_DATA_PATH.exists():
        print("\n  No RAG eval dataset found; expected "
              f"{RAG_EVAL_DATA_PATH}. Skipping RAG evaluation.\n")
        return None

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"\n  Could not import ragas or datasets ({exc}); skipping RAG evaluation.\n")
        return None

    raw = pd.read_json(RAG_EVAL_DATA_PATH, lines=True)
    if "model" not in raw.columns:
        raw["model"] = "rag_model"

    rows = []
    for model_name in raw["model"].unique():
        sub = raw[raw["model"] == model_name].reset_index(drop=True)
        if sub.empty:
            continue
        dataset = Dataset.from_pandas(sub)
        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
        except Exception as exc:  # pragma: no cover - optional external service
            print(
                "\n  RAG evaluation failed (likely missing or invalid LLM API key); "
                f"skipping RAG evaluation. Details: {exc}\n"
            )
            return None
        scores = {k: float(v) for k, v in result.items()}
        scores["model"] = model_name
        rows.append(scores)

    if not rows:
        print("  RAG dataset present but produced no rows; skipping.\n")
        return None

    rag_df = pd.DataFrame(rows)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rag_df.to_csv(RAG_EVAL_RESULTS_PATH, index=False)
    print(f"  RAG evaluation results saved to {RAG_EVAL_RESULTS_PATH}\n")
    return rag_df


def _plot_rag_radar(rag_df: pd.DataFrame, filename: Path) -> None:
    """
    Create a radar (spiderweb) plot over RAG metrics for each model.

    Metrics:
      - faithfulness
      - answer_relevancy
      - context_precision
      - context_recall
    """
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    for m in metrics:
        if m not in rag_df.columns:
            print(f"  RAG metric {m} missing; cannot create radar plot.")
            return

    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for _, row in rag_df.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model"])
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.0, 1.0)
    plt.title("RAG evaluation (ragas metrics)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  RAG radar plot saved: {filename}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import gc

    print("=" * 80)
    print("  StudyLens  -  Summary Evaluation")
    print("=" * 80)

    triples = discover_evaluation_triples()
    print(f"\nFound {len(triples)} (source, reference, summary) triples to evaluate.\n")

    if not triples:
        print("No files found. Check data/processed/, data/reference/, and data/outputs/ directories.")
        return

    # Unique tag for this run (used in plot filenames so runs don't overwrite each other).
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Cache source + reference text (avoid re-reading)
    source_cache: Dict[str, str] = {}
    reference_cache: Dict[str, str] = {}
    texts = []  # list of (source, reference, summary) aligned with triples
    for triple in triples:
        topic_key = triple["topic_key"]
        if topic_key not in source_cache:
            source_cache[topic_key] = triple["source_path"].read_text(encoding="utf-8")
        if topic_key not in reference_cache:
            reference_cache[topic_key] = triple["reference_path"].read_text(encoding="utf-8")

        source = source_cache[topic_key]
        reference = reference_cache[topic_key]
        summary = triple["summary_path"].read_text(encoding="utf-8")
        texts.append((source, reference, summary))

    if NLI_ONLY:
        # Load existing ROUGE+BERT results; run NLI only and merge back.
        csv_name = f"evaluation_results_{BATCH_SUFFIX}.csv" if BATCH_SUFFIX else "evaluation_results_all.csv"
        input_path = EVAL_DIR / csv_name
        if not input_path.exists():
            print(f"\n  NLI_ONLY: existing results not found: {input_path}")
            print("  Run a full eval (NLI_ONLY=False) first to generate the CSV.\n")
            return
        df = pd.read_csv(input_path)
        for col in ["entailment_ratio", "avg_entailment_score", "contradiction_ratio"]:
            if col not in df.columns:
                df[col] = float("nan")
        nli_indices = [i for i, t in enumerate(triples) if NLI_MODELS is None or t["model"] in NLI_MODELS]
        print("  NLI-only mode: ROUGE & BERTScore loaded from CSV. Running NLI only.\n")
        print("Phase 1/1: NLI factual consistency ...")
        print(f"  Running NLI for {len(nli_indices)} triples"
              + (f" (models: {', '.join(sorted(NLI_MODELS))})" if NLI_MODELS else " (all models)."))
        print(f"  Loading NLI model ({NLI_MODEL_NAME}) ...")
        nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        nli_tokenizer.model_max_length = 512
        nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        nli_model.eval()
        print("  NLI model loaded.\n")

        for k, i in enumerate(nli_indices, start=1):
            source, reference, summary = texts[i]
            nli = evaluate_nli(summary, source, nli_model=nli_model, nli_tokenizer=nli_tokenizer)
            file_name = triples[i]["summary_path"].name
            mask = df["file"] == file_name
            if mask.any():
                idx = df.index[mask].tolist()[0]
                df.loc[idx, "entailment_ratio"] = nli["entailment_ratio"]
                df.loc[idx, "avg_entailment_score"] = nli["avg_entailment_score"]
                df.loc[idx, "contradiction_ratio"] = nli["contradiction_ratio"]
            print(f"  [{k}/{len(nli_indices)}] {file_name[:45]}  "
                  f"entail={nli['avg_entailment_score']:.4f}  "
                  f"contradict={nli['contradiction_ratio']:.4f}")

        del nli_model, nli_tokenizer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("\n  NLI model freed from memory.\n")
        results = df.to_dict("records")
    else:
        # Full run: ROUGE, optional NLI, BERTScore.
        results = [
            {
                "topic":                 t["topic"],
                "topic_key":             t["topic_key"],
                "model":                 t["model"],
                "strategy":              t["strategy"],
                "ratio":                 t["ratio"],
                "file":                  t["summary_path"].name,
                "summary_words":         len(texts[i][2].split()),
                "entailment_ratio":      float("nan"),
                "avg_entailment_score":  float("nan"),
                "contradiction_ratio":   float("nan"),
            }
            for i, t in enumerate(triples)
        ]

        # ── Phase 1: ROUGE-L (no heavy model) ────────────────────────────────
        print("Phase 1/3: ROUGE-L (summary vs reference) ...")
        for i, (source, reference, summary) in enumerate(texts):
            rouge = evaluate_rouge_l(summary, reference)
            results[i].update(rouge)
            print(f"  [{i+1}/{len(triples)}] {results[i]['file'][:45]}  "
                  f"F1={rouge['rouge_l_f1']:.4f}")
        print()

        # ── Phase 2: NLI factual consistency (optional) ──────────────────────
        if RUN_NLI:
            nli_indices = [i for i, t in enumerate(triples) if NLI_MODELS is None or t["model"] in NLI_MODELS]
            print("Phase 2/3: NLI factual consistency ...")
            print(f"  Running NLI for {len(nli_indices)} triples"
                  + (f" (models: {', '.join(sorted(NLI_MODELS))})" if NLI_MODELS else " (all models)."))
            print(f"  Loading NLI model ({NLI_MODEL_NAME}) ...")
            nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
            nli_tokenizer.model_max_length = 512
            nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
            nli_model.eval()
            print("  NLI model loaded.\n")

            for k, i in enumerate(nli_indices, start=1):
                source, reference, summary = texts[i]
                nli = evaluate_nli(summary, source, nli_model=nli_model, nli_tokenizer=nli_tokenizer)
                results[i].update(nli)
                print(f"  [{k}/{len(nli_indices)}] {results[i]['file'][:45]}  "
                      f"entail={nli['avg_entailment_score']:.4f}  "
                      f"contradict={nli['contradiction_ratio']:.4f}")

            del nli_model, nli_tokenizer
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("\n  NLI model freed from memory.\n")
        else:
            print("Phase 2/3: NLI skipped (RUN_NLI=False).\n")

        # ── Phase 3: BERTScore ───────────────────────────────────────────────
        print("Phase 3/3: BERTScore (summary vs reference) ...")
        for i, (source, reference, summary) in enumerate(texts):
            bscore = evaluate_bertscore(summary, reference)
            results[i].update(bscore)
            print(f"  [{i+1}/{len(triples)}] {results[i]['file'][:45]}  "
                  f"F1={bscore['bertscore_f1']:.4f}")

        df = pd.DataFrame(results)

    # Print per-file summary
    print("\n" + "-" * 60)
    for r in results:
        nli_val = r.get("avg_entailment_score", float("nan"))
        print(f"  {r['file'][:45]}  ROUGE={r['rouge_l_f1']:.4f}  "
              f"NLI={nli_val:.4f}  "
              f"BERT={r['bertscore_f1']:.4f}")

    # ── Results table ────────────────────────────────────────────────────
    df = pd.DataFrame(results)

    display_cols = [
        "file",
        "topic",
        "model",
        "strategy",
        "ratio",
        "summary_words",
        "rouge_l_f1",
        "bertscore_f1",
    ]
    optional_nli = ["avg_entailment_score", "contradiction_ratio"]
    for c in optional_nli:
        if c in df.columns:
            display_cols.insert(display_cols.index("bertscore_f1"), c)
    print("\n" + "=" * 80)
    print("  FULL RESULTS")
    print("=" * 80)
    print(df[display_cols].to_string(index=False))

    # ── Per-topic breakdown ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PER-TOPIC BREAKDOWN")
    print("=" * 80)
    for topic in df["topic"].unique():
        sub = df[df["topic"] == topic]
        print(f"\n  {topic}:")
        topic_cols = ["file", "model", "strategy", "ratio", "rouge_l_f1", "bertscore_f1"]
        for c in ["avg_entailment_score", "contradiction_ratio"]:
            if c in sub.columns:
                topic_cols.insert(topic_cols.index("bertscore_f1"), c)
        print(sub[topic_cols].to_string(index=False))

    # ── Best technique (averaged across topics) ──────────────────────────
    if RUN_NLI:
        metric_cols = ["rouge_l_f1", "avg_entailment_score", "bertscore_f1"]
    else:
        metric_cols = ["rouge_l_f1", "bertscore_f1"]

    avg = df.groupby(["model", "strategy", "ratio"])[metric_cols].mean().round(4)
    avg["composite"] = avg[metric_cols].mean(axis=1).round(4)
    avg = avg.sort_values("composite", ascending=False)

    print("\n" + "=" * 80)
    print("  AVERAGE ACROSS TOPICS  (higher = better)")
    print("=" * 80)
    print(avg.to_string())

    best_idx = avg["composite"].idxmax()
    best_model, best_strategy, best_ratio = best_idx
    best_score = avg.loc[best_idx, "composite"]

    ratio_str = f", ratio={best_ratio}" if best_ratio else ""
    print(f"\n  BEST TECHNIQUE:  {best_model} ({best_strategy}{ratio_str})")
    print(f"  Composite score: {best_score:.4f}")
    print(f"    ROUGE-L F1        = {avg.loc[best_idx, 'rouge_l_f1']:.4f}")
    if "avg_entailment_score" in avg.columns:
        print(f"    NLI Entailment    = {avg.loc[best_idx, 'avg_entailment_score']:.4f}")
    print(f"    BERTScore F1      = {avg.loc[best_idx, 'bertscore_f1']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    # Save metrics table for downstream analysis / plotting.
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = f"evaluation_results_{BATCH_SUFFIX}.csv" if BATCH_SUFFIX else "evaluation_results_all.csv"
    output_path = EVAL_DIR / csv_name
    df.to_csv(output_path, index=False)
    print(f"\n  Results saved to {output_path}")

    # ── Visualizations ───────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # Include both batch suffix (if any) and a unique run tag so plots do not overwrite.
    if BATCH_SUFFIX:
        plot_suffix = f"_{BATCH_SUFFIX}_{run_tag}"
    else:
        plot_suffix = f"_{run_tag}"
    _plot_metric_bars(df, metric="rouge_l_f1", ylabel="ROUGE-L F1",
                      filename=PLOTS_DIR / f"rouge_l_f1_by_bmodel_ratio{plot_suffix}.png")
    _plot_metric_bars(df, metric="bertscore_f1", ylabel="BERTScore F1",
                      filename=PLOTS_DIR / f"bertscore_f1_by_bmodel_ratio{plot_suffix}.png")
    if "avg_entailment_score" in df.columns:
        _plot_metric_bars(df, metric="avg_entailment_score", ylabel="NLI entailment score",
                          filename=PLOTS_DIR / f"nli_entailment_by_bmodel_ratio{plot_suffix}.png")

    # ── Optional: RAG evaluation and radar plot ──────────────────────────
    rag_df = _run_rag_evaluation()
    if rag_df is not None and not rag_df.empty:
        rag_plot = PLOTS_DIR / f"rag_radar_plot{plot_suffix}.png" if plot_suffix else PLOTS_DIR / "rag_radar_plot.png"
        _plot_rag_radar(rag_df, filename=rag_plot)


if __name__ == "__main__":
    main()
