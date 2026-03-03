"""
eval.py - Evaluation of generated summaries using three metrics.

Metric 1  ROUGE-L     Surface-level recall of source content (longest common subsequence).
Metric 2  NLI         Factual consistency via roberta-large-mnli entailment checking.
Metric 3  BERTScore   Semantic similarity using contextual embeddings.

Compares BART vs LongT5, concat vs final strategies, and identifies the best technique.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt_tab", quiet=True)


# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
SOURCE_DIR  = ROOT_DIR / "data" / "processed"
SUMMARY_DIR = ROOT_DIR / "data" / "outputs"

# Topics (keys match filenames)
TOPICS = {
    "dl_s6_neural_networks_nlp":    "Neural Networks for NLP",
    "dl_s7_attention_transformers":  "Attention & Transformers",
}

# Summary variants: {model}_{strategy}
VARIANTS = ["bart_final", "longt5_final", "bart_concat", "longt5_concat"]

# NLI settings
NLI_MODEL_NAME     = "roberta-large-mnli"
NLI_PREMISE_TOKENS = 400   # max tokens per source chunk (leaves room for hypothesis in 512 limit)
NLI_TOP_K          = 5     # only check top-k most relevant source chunks per sentence


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

    for hyp in summary_sents:
        top_chunks = _find_top_chunks(hyp, source_chunks, NLI_TOP_K)

        max_entail = 0.0
        is_contradicted = False

        for premise in top_chunks:
            inputs = nli_tokenizer(
                premise, hyp,
                return_tensors="pt", truncation=True, max_length=512,
            )
            with torch.no_grad():
                logits = nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            # roberta-large-mnli classes: 0=contradiction, 1=neutral, 2=entailment
            max_entail = max(max_entail, probs[2].item())
            if probs[0].item() > 0.5:
                is_contradicted = True

        entailment_scores.append(max_entail)
        contradiction_flags.append(is_contradicted)

    n = len(entailment_scores)
    return {
        "entailment_ratio":     round(sum(1 for s in entailment_scores if s > 0.5) / n, 4),
        "avg_entailment_score": round(float(np.mean(entailment_scores)), 4),
        "contradiction_ratio":  round(sum(contradiction_flags) / n, 4),
    }


# ── Metric 3: BERTScore ─────────────────────────────────────────────────

def evaluate_bertscore(summary: str, reference: str) -> Dict[str, float]:
    """
    BERTScore: token-level semantic similarity using contextual embeddings.
    Uses roberta-large with baseline rescaling for calibrated scores.
    Note: reference is truncated to 512 tokens by the model; this is standard
    for BERTScore evaluation of summaries against long source documents.
    """
    P, R, F1 = bert_score_fn(
        [summary], [reference],
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )
    return {
        "bertscore_precision": round(P[0].item(), 4),
        "bertscore_recall":    round(R[0].item(), 4),
        "bertscore_f1":        round(F1[0].item(), 4),
    }


# ── File discovery ───────────────────────────────────────────────────────

def discover_evaluation_pairs() -> List[Dict]:
    """Find all (source, summary) file pairs based on naming convention."""
    pairs = []
    for topic_key, topic_name in TOPICS.items():
        source_path = SOURCE_DIR / f"{topic_key}_ori.txt"
        if not source_path.exists():
            print(f"  WARNING: Source not found: {source_path}")
            continue

        for variant in VARIANTS:
            summary_path = SUMMARY_DIR / f"{topic_key}_sum_{variant}.txt"
            if not summary_path.exists():
                print(f"  WARNING: Summary not found: {summary_path}")
                continue

            model_name, strategy = variant.rsplit("_", 1)
            pairs.append({
                "topic":        topic_name,
                "topic_key":    topic_key,
                "model":        model_name,
                "strategy":     strategy,
                "source_path":  source_path,
                "summary_path": summary_path,
            })
    return pairs


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import gc

    print("=" * 80)
    print("  StudyLens  -  Summary Evaluation")
    print("=" * 80)

    pairs = discover_evaluation_pairs()
    print(f"\nFound {len(pairs)} summary files to evaluate.\n")

    if not pairs:
        print("No files found. Check data/processed/ and data/outputs/ directories.")
        return

    # Cache source + summary text (avoid re-reading)
    source_cache: Dict[str, str] = {}
    texts = []     # list of (source, summary) aligned with pairs
    for pair in pairs:
        topic_key = pair["topic_key"]
        if topic_key not in source_cache:
            source_cache[topic_key] = pair["source_path"].read_text(encoding="utf-8")
        source = source_cache[topic_key]
        summary = pair["summary_path"].read_text(encoding="utf-8")
        texts.append((source, summary))

    # Initialize result rows
    results = [{
        "topic":         p["topic"],
        "model":         p["model"],
        "strategy":      p["strategy"],
        "file":          p["summary_path"].name,
        "summary_words": len(texts[i][1].split()),
    } for i, p in enumerate(pairs)]

    # ── Phase 1: ROUGE-L (no heavy model) ────────────────────────────────
    print("Phase 1/3: ROUGE-L ...")
    for i, (source, summary) in enumerate(texts):
        rouge = evaluate_rouge_l(summary, source)
        results[i].update(rouge)
        print(f"  [{i+1}/{len(pairs)}] {results[i]['file'][:45]}  "
              f"F1={rouge['rouge_l_f1']:.4f}")
    print()

    # ── Phase 2: NLI factual consistency ─────────────────────────────────
    print("Phase 2/3: NLI factual consistency ...")
    print("  Loading NLI model (roberta-large-mnli) ...")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    nli_model.eval()
    print("  NLI model loaded.\n")

    for i, (source, summary) in enumerate(texts):
        nli = evaluate_nli(summary, source, nli_model=nli_model, nli_tokenizer=nli_tokenizer)
        results[i].update(nli)
        print(f"  [{i+1}/{len(pairs)}] {results[i]['file'][:45]}  "
              f"entail={nli['avg_entailment_score']:.4f}  "
              f"contradict={nli['contradiction_ratio']:.4f}")

    # Free NLI model before loading BERTScore
    del nli_model, nli_tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\n  NLI model freed from memory.\n")

    # ── Phase 3: BERTScore ───────────────────────────────────────────────
    print("Phase 3/3: BERTScore ...")
    for i, (source, summary) in enumerate(texts):
        bscore = evaluate_bertscore(summary, source)
        results[i].update(bscore)
        print(f"  [{i+1}/{len(pairs)}] {results[i]['file'][:45]}  "
              f"F1={bscore['bertscore_f1']:.4f}")

    # Print per-file summary
    print("\n" + "-" * 60)
    for r in results:
        print(f"  {r['file'][:45]}  ROUGE={r['rouge_l_f1']:.4f}  "
              f"NLI={r['avg_entailment_score']:.4f}  "
              f"BERT={r['bertscore_f1']:.4f}")

    # ── Results table ────────────────────────────────────────────────────
    df = pd.DataFrame(results)

    display_cols = [
        "file", "topic", "model", "strategy", "summary_words",
        "rouge_l_f1", "avg_entailment_score", "contradiction_ratio", "bertscore_f1",
    ]
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
        print(sub[["file", "model", "strategy", "rouge_l_f1", "avg_entailment_score",
                    "contradiction_ratio", "bertscore_f1"]].to_string(index=False))

    # ── Best technique (averaged across topics) ──────────────────────────
    metric_cols = ["rouge_l_f1", "avg_entailment_score", "bertscore_f1"]
    avg = df.groupby(["model", "strategy"])[metric_cols].mean().round(4)
    avg["composite"] = avg[metric_cols].mean(axis=1).round(4)
    avg = avg.sort_values("composite", ascending=False)

    print("\n" + "=" * 80)
    print("  AVERAGE ACROSS TOPICS  (higher = better)")
    print("=" * 80)
    print(avg.to_string())

    best_idx = avg["composite"].idxmax()
    best_model, best_strategy = best_idx
    best_score = avg.loc[best_idx, "composite"]

    print(f"\n  BEST TECHNIQUE:  {best_model} ({best_strategy})")
    print(f"  Composite score: {best_score:.4f}")
    print(f"    ROUGE-L F1        = {avg.loc[best_idx, 'rouge_l_f1']:.4f}")
    print(f"    NLI Entailment    = {avg.loc[best_idx, 'avg_entailment_score']:.4f}")
    print(f"    BERTScore F1      = {avg.loc[best_idx, 'bertscore_f1']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    output_path = SUMMARY_DIR / "evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
