"""
eval.py - Unified evaluation of ALL generated summaries.

Auto-discovers every summary .txt file under data/outputs/ and computes:
  Metric 1  ROUGE-L     Surface-level overlap vs. reference summaries.
  Metric 2  NLI         Factual consistency vs. original slide text. (optional, --run-nli)
  Metric 3  BERTScore   Semantic similarity vs. reference summaries.

Output directory: data/eval/ (sibling to data/outputs/, NOT inside it).
Generates two CSVs:
  - evaluation_results_all.csv   (per-file metrics)
  - evaluation_averages.csv      (per-model averages)

Usage:
    python scripts/eval.py                          # ROUGE + BERTScore only
    python scripts/eval.py --run-nli                # also run NLI (slower)
    python scripts/eval.py --models bart longt5     # only eval specific models

Original eval logic by Sharmil; unified auto-discovery by Sophia.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import gc
import re
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
ROOT_DIR      = Path(__file__).resolve().parent.parent
SOURCE_DIR    = ROOT_DIR / "data" / "processed"
REFERENCE_DIR = ROOT_DIR / "data" / "reference"
SUMMARY_DIR   = ROOT_DIR / "data" / "outputs"
EVAL_DIR      = ROOT_DIR / "data" / "eval"           # outside outputs
PLOTS_DIR     = EVAL_DIR / "plots"

# RAG evaluation paths
RAG_EVAL_DATA_PATH    = EVAL_DIR / "rag_eval_data.jsonl"
RAG_EVAL_RESULTS_PATH = EVAL_DIR / "rag_eval_results.csv"

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

ALL_TOPIC_KEYS = set(TOPIC_NAMES.keys())

# NLI settings
NLI_MODEL_NAME     = "cross-encoder/nli-distilroberta-base"
NLI_PREMISE_TOKENS = 400
NLI_TOP_K          = 5
MAX_TOKENS_FOR_SCORE = 450


# ════════════════════════════════════════════════════════════════════════
# Metrics (unchanged from Sharmil's original implementations)
# ════════════════════════════════════════════════════════════════════════

def evaluate_rouge_l(summary: str, reference: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {
        "rouge_l_precision": round(scores["rougeL"].precision, 4),
        "rouge_l_recall":    round(scores["rougeL"].recall, 4),
        "rouge_l_f1":        round(scores["rougeL"].fmeasure, 4),
    }


def _build_source_chunks(source: str, tokenizer, max_tokens: int) -> List[str]:
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
    hyp_words = set(hypothesis.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        union = hyp_words | chunk_words
        scores.append(len(hyp_words & chunk_words) / len(union) if union else 0.0)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def evaluate_nli(summary: str, source: str, nli_model, nli_tokenizer) -> Dict[str, float]:
    summary_sents = sent_tokenize(summary)
    if not summary_sents:
        return {"entailment_ratio": 0.0, "avg_entailment_score": 0.0, "contradiction_ratio": 0.0}

    source_chunks = _build_source_chunks(source, nli_tokenizer, NLI_PREMISE_TOKENS)
    entailment_scores = []
    contradiction_flags = []
    hyp_max_words = 100

    for hyp in summary_sents:
        hyp_chunks = _chunk_text_by_max_words(hyp, max_words=hyp_max_words)
        if not hyp_chunks:
            hyp_chunks = [" ".join(hyp.split()[:hyp_max_words])]

        sent_entailments = []
        sent_contradicted = False

        for hyp_part in hyp_chunks:
            top_chunks = _find_top_chunks(hyp_part, source_chunks, NLI_TOP_K)
            max_entail = 0.0
            for premise in top_chunks:
                inputs = nli_tokenizer(
                    premise, hyp_part,
                    return_tensors="pt", truncation="longest_first",
                    max_length=512, padding=False,
                )
                with torch.no_grad():
                    logits = nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                max_entail = max(max_entail, probs[1].item())
                if probs[0].item() > 0.5:
                    sent_contradicted = True
            sent_entailments.append(max_entail)

        entailment_scores.append(float(np.mean(sent_entailments)))
        contradiction_flags.append(sent_contradicted)

    n = len(entailment_scores)
    return {
        "entailment_ratio":     round(sum(1 for s in entailment_scores if s > 0.5) / n, 4),
        "avg_entailment_score": round(float(np.mean(entailment_scores)), 4),
        "contradiction_ratio":  round(sum(contradiction_flags) / n, 4),
    }


def _chunk_text_by_max_words(text: str, max_words: int = MAX_TOKENS_FOR_SCORE) -> List[str]:
    sentences = sent_tokenize(text)
    chunks, buf, buf_words = [], [], 0
    for sent in sentences:
        sw = len(sent.split())
        if sw > max_words:
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_words = [], 0
            words = sent.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue
        if buf_words + sw > max_words and buf:
            chunks.append(" ".join(buf))
            buf, buf_words = [sent], sw
        else:
            buf.append(sent)
            buf_words += sw
    if buf:
        chunks.append(" ".join(buf))
    return chunks if chunks else [text.strip() or " "]


def evaluate_bertscore(summary: str, reference: str) -> Dict[str, float]:
    ref_chunks = _chunk_text_by_max_words(reference)
    sum_chunks = _chunk_text_by_max_words(summary)
    n = min(len(ref_chunks), len(sum_chunks))
    if n == 0:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    P, R, F1 = bert_score_fn(
        sum_chunks[:n], ref_chunks[:n],
        lang="en", rescale_with_baseline=True, verbose=False,
    )
    return {
        "bertscore_precision": round(float(P.mean().item()), 4),
        "bertscore_recall":    round(float(R.mean().item()), 4),
        "bertscore_f1":        round(float(F1.mean().item()), 4),
    }


# ════════════════════════════════════════════════════════════════════════
# Auto-discovery
# ════════════════════════════════════════════════════════════════════════

def _parse_summary_path(path: Path) -> Optional[Dict]:
    """Parse a summary file path to extract topic_key, model, strategy, ratio."""
    try:
        rel = path.relative_to(SUMMARY_DIR)
    except ValueError:
        return None

    parts = rel.parts
    fname = path.name

    topic_match = re.match(r"^((?:dl|ml)_s\d+)", fname)
    if not topic_match:
        return None
    topic_key = topic_match.group(1)
    if topic_key not in ALL_TOPIC_KEYS:
        return None
    if len(parts) < 2:
        return None

    category = parts[0]

    if category == "naive":
        strategy = parts[1] if len(parts) >= 2 else "unknown"
        return {"topic_key": topic_key, "model": f"naive_{strategy}",
                "strategy": strategy, "ratio": None}

    elif category == "classical_ml":
        strategy = parts[1] if len(parts) >= 2 else "unknown"
        return {"topic_key": topic_key, "model": "tfidf",
                "strategy": strategy, "ratio": None}

    elif category == "neural_network":
        if len(parts) < 3:
            return None
        model_name = parts[1]
        sub = parts[2]
        ratio = None
        strategy = sub
        if sub.startswith("ratio"):
            ratio = sub
            strategy = "ratio"
        return {"topic_key": topic_key, "model": model_name,
                "strategy": strategy, "ratio": ratio}

    elif category == "finetune":
        if len(parts) < 3:
            return None
        return {"topic_key": topic_key, "model": parts[1],
                "strategy": parts[2], "ratio": None}

    return None


def discover_all_summaries(model_filter: Optional[List[str]] = None) -> List[Dict]:
    """Recursively find all .txt summary files under data/outputs/."""
    triples = []
    for txt_path in sorted(SUMMARY_DIR.rglob("*.txt")):
        if "eval" in txt_path.parts:
            continue
        parsed = _parse_summary_path(txt_path)
        if parsed is None:
            continue
        topic_key = parsed["topic_key"]
        ref_path = REFERENCE_DIR / f"{topic_key}_ref.txt"
        source_path = SOURCE_DIR / f"{topic_key}_ori.txt"
        if not ref_path.exists() or not source_path.exists():
            continue
        if model_filter and parsed["model"] not in model_filter:
            continue
        triples.append({
            "topic":          TOPIC_NAMES.get(topic_key, topic_key),
            "topic_key":      topic_key,
            "model":          parsed["model"],
            "strategy":       parsed["strategy"],
            "ratio":          parsed["ratio"],
            "source_path":    source_path,
            "reference_path": ref_path,
            "summary_path":   txt_path,
        })
    return triples


# ════════════════════════════════════════════════════════════════════════
# Plotting helpers (from Sharmil's original)
# ════════════════════════════════════════════════════════════════════════

def _plot_metric_bars(df: pd.DataFrame, metric: str, ylabel: str,
                      filename: Path, plot_models: Optional[set] = None) -> None:
    if metric not in df.columns:
        return
    plot_df = df if plot_models is None else df[df["model"].isin(plot_models)]
    if metric in {"entailment_ratio", "avg_entailment_score", "contradiction_ratio"}:
        plot_df = plot_df.dropna(subset=[metric])
        if plot_df.empty:
            return
    agg = plot_df.groupby(["model", "ratio"], dropna=False)[metric].mean().reset_index()
    models = agg["model"].unique()
    ratios = [r for r in agg["ratio"].unique() if isinstance(r, str)] or [None]
    x = np.arange(len(models))
    width = 0.15 if len(ratios) > 1 else 0.6
    plt.figure(figsize=(10, 5))
    for idx, ratio in enumerate(ratios):
        sub = agg[agg["ratio"] == ratio] if ratio is not None else agg[agg["ratio"].isna()]
        heights = [sub[sub["model"] == m][metric].iloc[0]
                    if not sub[sub["model"] == m].empty else 0.0 for m in models]
        offsets = x + (idx - (len(ratios) - 1) / 2) * width
        plt.bar(offsets, heights, width=width, label=ratio or "no-ratio")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by model and ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Plot saved: {filename}")


# ════════════════════════════════════════════════════════════════════════
# RAG evaluation (from Sharmil's original, unchanged)
# ════════════════════════════════════════════════════════════════════════

def _run_rag_evaluation() -> Optional[pd.DataFrame]:
    if not RAG_EVAL_DATA_PATH.exists():
        print(f"\n  No RAG eval dataset found at {RAG_EVAL_DATA_PATH}. Skipping.\n")
        return None
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
    except Exception as exc:
        print(f"\n  Could not import ragas/datasets ({exc}); skipping RAG evaluation.\n")
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
                metrics=[faithfulness, answer_relevancy,
                         context_precision, context_recall],
            )
        except Exception as exc:
            print(f"\n  RAG evaluation failed: {exc}\n")
            return None
        scores = {k: float(v) for k, v in result.items()}
        scores["model"] = model_name
        rows.append(scores)

    if not rows:
        return None
    rag_df = pd.DataFrame(rows)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rag_df.to_csv(RAG_EVAL_RESULTS_PATH, index=False)
    print(f"  RAG evaluation results saved to {RAG_EVAL_RESULTS_PATH}\n")
    return rag_df


def _plot_rag_radar(rag_df: pd.DataFrame, filename: Path) -> None:
    metrics = ["faithfulness", "answer_relevancy",
               "context_precision", "context_recall"]
    for m in metrics:
        if m not in rag_df.columns:
            return
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    for _, row in rag_df.iterrows():
        values = [row[m] for m in metrics] + [row[metrics[0]]]
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


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="StudyLens Evaluation")
    parser.add_argument("--run-nli", action="store_true",
                        help="Also compute NLI factual consistency (slower)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only evaluate these models (e.g. --models bart longt5)")
    args = parser.parse_args()

    print("=" * 70)
    print("  StudyLens - Summary Evaluation")
    print("=" * 70)

    triples = discover_all_summaries(model_filter=args.models)
    print(f"\nDiscovered {len(triples)} summary files to evaluate.\n")

    if not triples:
        print("No files found. Check data/outputs/ directory.")
        return

    # Discovery summary
    model_counts = {}
    for t in triples:
        key = f"{t['model']}/{t['strategy']}"
        if t["ratio"]:
            key += f"/{t['ratio']}"
        model_counts[key] = model_counts.get(key, 0) + 1
    for k, v in sorted(model_counts.items()):
        print(f"  {k}: {v} files")
    print()

    # Read texts
    source_cache, reference_cache = {}, {}
    texts = []
    for t in triples:
        tk = t["topic_key"]
        if tk not in source_cache:
            source_cache[tk] = t["source_path"].read_text(encoding="utf-8")
        if tk not in reference_cache:
            reference_cache[tk] = t["reference_path"].read_text(encoding="utf-8")
        summary = t["summary_path"].read_text(encoding="utf-8")
        texts.append((source_cache[tk], reference_cache[tk], summary))

    # Initialize results
    results = []
    for i, t in enumerate(triples):
        results.append({
            "topic":                t["topic"],
            "topic_key":            t["topic_key"],
            "model":                t["model"],
            "strategy":             t["strategy"],
            "ratio":                t["ratio"],
            "file":                 t["summary_path"].name,
            "summary_words":        len(texts[i][2].split()),
            "entailment_ratio":     float("nan"),
            "avg_entailment_score": float("nan"),
            "contradiction_ratio":  float("nan"),
        })

    # Phase 1: ROUGE-L
    print("Phase 1/3: ROUGE-L ...")
    for i, (source, reference, summary) in enumerate(texts):
        rouge = evaluate_rouge_l(summary, reference)
        results[i].update(rouge)
        if (i + 1) % 20 == 0 or i == len(texts) - 1:
            print(f"  [{i+1}/{len(texts)}] done")
    print()

    # Phase 2: NLI (optional)
    if args.run_nli:
        print("Phase 2/3: NLI factual consistency ...")
        print(f"  Loading NLI model ({NLI_MODEL_NAME}) ...")
        nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        nli_tokenizer.model_max_length = 512
        nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        nli_model.eval()
        for i, (source, reference, summary) in enumerate(texts):
            nli = evaluate_nli(summary, source, nli_model, nli_tokenizer)
            results[i].update(nli)
            if (i + 1) % 10 == 0 or i == len(texts) - 1:
                print(f"  [{i+1}/{len(texts)}] done")
        del nli_model, nli_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()
    else:
        print("Phase 2/3: NLI skipped (use --run-nli to enable)\n")

    # Phase 3: BERTScore
    print("Phase 3/3: BERTScore ...")
    for i, (source, reference, summary) in enumerate(texts):
        bscore = evaluate_bertscore(summary, reference)
        results[i].update(bscore)
        if (i + 1) % 20 == 0 or i == len(texts) - 1:
            print(f"  [{i+1}/{len(texts)}] done")
    print()

    # ── Save ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    per_file_path = EVAL_DIR / "evaluation_results_all.csv"
    df.to_csv(per_file_path, index=False)
    print(f"Per-file results saved to {per_file_path}")

    # Averages CSV
    metric_cols = ["rouge_l_f1", "bertscore_f1"]
    if args.run_nli:
        metric_cols = ["rouge_l_f1", "avg_entailment_score", "bertscore_f1"]
    avg = (df.groupby(["model", "strategy", "ratio"], dropna=False)[metric_cols]
           .mean().round(4))
    avg["composite"] = avg[metric_cols].mean(axis=1).round(4)
    avg = avg.sort_values("composite", ascending=False)
    avg_path = EVAL_DIR / "evaluation_averages.csv"
    avg.to_csv(avg_path)
    print(f"Average results saved to {avg_path}")

    # Print averages
    print("\n" + "=" * 70)
    print("  AVERAGES ACROSS TOPICS (higher = better)")
    print("=" * 70)
    print(avg.to_string())
    best_idx = avg["composite"].idxmax()
    best_model, best_strat, best_ratio = best_idx
    ratio_str = f", ratio={best_ratio}" if best_ratio else ""
    print(f"\n  BEST: {best_model} ({best_strat}{ratio_str})  "
          f"composite={avg.loc[best_idx, 'composite']:.4f}")

    # Plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    _plot_metric_bars(df, "rouge_l_f1", "ROUGE-L F1",
                      PLOTS_DIR / f"rouge_l_f1_{tag}.png")
    _plot_metric_bars(df, "bertscore_f1", "BERTScore F1",
                      PLOTS_DIR / f"bertscore_f1_{tag}.png")
    if args.run_nli:
        _plot_metric_bars(df, "avg_entailment_score", "NLI Entailment",
                          PLOTS_DIR / f"nli_entailment_{tag}.png")

    # RAG
    rag_df = _run_rag_evaluation()
    if rag_df is not None and not rag_df.empty:
        _plot_rag_radar(rag_df, PLOTS_DIR / f"rag_radar_{tag}.png")

    print(f"\nTotal: {len(df)} files evaluated")


if __name__ == "__main__":
    main()