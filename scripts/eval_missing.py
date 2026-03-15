"""
eval_missing.py - Evaluate summaries not covered by the main eval.py

Discovers additional model outputs (longt5, bart-samsum, bart final,
naive random, qwen7b-ft) and computes ROUGE-L + BERTScore against
reference summaries. Appends results to the existing CSV.

Usage (base environment, CPU is fine):
    python scripts/eval_missing.py

Or on Colab:
    !python scripts/eval_missing.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt_tab", quiet=True)

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
SOURCE_DIR    = ROOT_DIR / "data" / "processed"
REFERENCE_DIR = ROOT_DIR / "data" / "reference"
SUMMARY_DIR   = ROOT_DIR / "data" / "outputs"
EVAL_DIR      = SUMMARY_DIR / "eval"

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

# ── Define what to evaluate ──────────────────────────────────────────────
# Each entry: (model_name, strategy, ratio, path_pattern)
# path_pattern uses {topic} as placeholder
ADDITIONAL_MODELS = [
    # BART final
    ("bart", "final", None,
     "neural_network/bart/final/{topic}_sum_bart_final.txt"),
    # Long-T5 concat
    ("longt5", "concat", None,
     "neural_network/longt5/concat/{topic}_sum_longt5_concat.txt"),
    # Long-T5 final
    ("longt5", "final", None,
     "neural_network/longt5/final/{topic}_sum_longt5_final.txt"),
    # BART-SAMSum concat
    ("bart-samsum", "concat", None,
     "neural_network/bart-samsum/concat/{topic}_sum_bart-samsum_concat.txt"),
    # BART-SAMSum final
    ("bart-samsum", "final", None,
     "neural_network/bart-samsum/final/{topic}_sum_bart-samsum_final.txt"),
    # Naive random
    ("naive_random", "random", None,
     "naive/random/{topic}_sum_naive_random.txt"),
    # Qwen7b fine-tuned
    ("qwen7b-ft", "final", None,
     "finetune/qwen7b-ft/final/{topic}_sum_qwen7b-ft_final.txt"),
]

MAX_TOKENS_FOR_SCORE = 450


# ── Metrics ──────────────────────────────────────────────────────────────

def evaluate_rouge_l(summary: str, reference: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {
        "rouge_l_precision": round(scores["rougeL"].precision, 4),
        "rouge_l_recall":    round(scores["rougeL"].recall, 4),
        "rouge_l_f1":        round(scores["rougeL"].fmeasure, 4),
    }


def _chunk_text_by_max_words(text: str, max_words: int = MAX_TOKENS_FOR_SCORE) -> list:
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
                chunks.append(" ".join(words[i:i+max_words]))
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


# ── Discovery ────────────────────────────────────────────────────────────

def discover_missing_triples() -> list:
    """Find all (model, topic) pairs that have output files."""
    topics = sorted([p.stem.replace("_ori", "") for p in SOURCE_DIR.glob("*_ori.txt")])
    triples = []

    for model_name, strategy, ratio, pattern in ADDITIONAL_MODELS:
        for topic_id in topics:
            summary_path = SUMMARY_DIR / pattern.format(topic=topic_id)
            ref_path = REFERENCE_DIR / f"{topic_id}_ref.txt"
            source_path = SOURCE_DIR / f"{topic_id}_ori.txt"

            if not summary_path.exists():
                continue
            if not ref_path.exists():
                continue

            triples.append({
                "topic":          TOPIC_NAMES.get(topic_id, topic_id),
                "topic_key":      topic_id,
                "model":          model_name,
                "strategy":       strategy,
                "ratio":          ratio,
                "source_path":    source_path,
                "reference_path": ref_path,
                "summary_path":   summary_path,
            })

    return triples


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  StudyLens - Supplementary Evaluation (missing models)")
    print("=" * 70)

    triples = discover_missing_triples()
    print(f"\nFound {len(triples)} new summary files to evaluate.\n")

    if not triples:
        print("Nothing to evaluate. Check that output files exist.")
        return

    # Show what we found
    for t in triples:
        print(f"  {t['summary_path'].name}  [{t['model']} / {t['strategy']}]")
    print()

    results = []

    for i, t in enumerate(triples):
        reference = t["reference_path"].read_text(encoding="utf-8")
        summary = t["summary_path"].read_text(encoding="utf-8")

        print(f"[{i+1}/{len(triples)}] {t['summary_path'].name}")

        # ROUGE-L
        rouge = evaluate_rouge_l(summary, reference)

        # BERTScore
        bscore = evaluate_bertscore(summary, reference)

        print(f"  ROUGE-L F1={rouge['rouge_l_f1']:.4f}  "
              f"BERTScore F1={bscore['bertscore_f1']:.4f}")

        row = {
            "topic":                t["topic"],
            "topic_key":            t["topic_key"],
            "model":                t["model"],
            "strategy":             t["strategy"],
            "ratio":                t["ratio"],
            "file":                 t["summary_path"].name,
            "summary_words":        len(summary.split()),
            "entailment_ratio":     float("nan"),
            "avg_entailment_score": float("nan"),
            "contradiction_ratio":  float("nan"),
        }
        row.update(rouge)
        row.update(bscore)
        results.append(row)

    new_df = pd.DataFrame(results)

    # Try to merge with existing CSV
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    existing_csv = EVAL_DIR / "evaluation_results_all.csv"

    if existing_csv.exists():
        old_df = pd.read_csv(existing_csv)
        # Remove any rows for models we're re-evaluating to avoid duplicates
        new_models = set(new_df["model"].unique())
        old_df = old_df[~old_df["model"].isin(new_models)]
        combined = pd.concat([old_df, new_df], ignore_index=True)
        print(f"\nMerged with existing CSV ({len(old_df)} old + {len(new_df)} new "
              f"= {len(combined)} total rows)")
    else:
        combined = new_df
        print(f"\nNo existing CSV found. Creating new with {len(combined)} rows.")

    output_path = EVAL_DIR / "evaluation_results_all.csv"
    combined.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  NEW RESULTS")
    print("=" * 70)
    display_cols = ["file", "model", "strategy", "summary_words",
                    "rouge_l_f1", "bertscore_f1"]
    print(new_df[display_cols].to_string(index=False))

    # Averages
    metric_cols = ["rouge_l_f1", "bertscore_f1"]
    avg = new_df.groupby(["model", "strategy"])[metric_cols].mean().round(4)
    avg["composite"] = avg[metric_cols].mean(axis=1).round(4)
    avg = avg.sort_values("composite", ascending=False)
    print("\n  AVERAGES:")
    print(avg.to_string())


if __name__ == "__main__":
    main()