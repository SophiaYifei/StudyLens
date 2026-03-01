"""
scripts/build_features.py
Chunking, embedding, topic concatenation, and output saving in data/processed folder.
"""

import re, json
from pathlib import Path
from typing import List, Dict, Callable

import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Defaults ───────────────────────────────────────────────────────────

CHUNK_SIZE_TOKENS   = 200
CHUNK_OVERLAP_SENTS = 1
MIN_CHUNK_TOKENS    = 30
TOKENIZER_MODEL     = "facebook/bart-large-cnn"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"


def make_token_counter(tokenizer) -> Callable[[str], int]:
    """Return a closure that counts BPE tokens for the given tokenizer."""
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return count_tokens


# ── Strategy 1: Sentence-aware sliding window ──────────────────────────
# Best for: transcripts (continuous prose, no natural section markers)

def chunk_by_sentences(
    text: str,
    count_tokens_fn: Callable[[str], int],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap_sents: int = CHUNK_OVERLAP_SENTS,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> List[str]:
    sentences = sent_tokenize(text)
    if not sentences:
        text_stripped = text.strip()
        if text_stripped and count_tokens_fn(text_stripped) >= min_tokens:
            return [text_stripped]
        return []

    chunks: List[str] = []
    cur_sents: List[str] = []
    cur_tokens: int = 0
    i = 0

    while i < len(sentences):
        sent = sentences[i]
        sent_tok = count_tokens_fn(sent)

        if sent_tok > chunk_size and not cur_sents:
            chunks.append(sent)
            i += 1
            continue

        if cur_tokens + sent_tok > chunk_size and cur_sents:
            chunk_text = " ".join(cur_sents)
            if count_tokens_fn(chunk_text) >= min_tokens:
                chunks.append(chunk_text)

            if overlap_sents and len(cur_sents) > overlap_sents:
                ov = cur_sents[-overlap_sents:]
                ov_tok = sum(count_tokens_fn(s) for s in ov)
                if ov_tok < chunk_size:
                    cur_sents, cur_tokens = ov, ov_tok
                else:
                    cur_sents, cur_tokens = [], 0
            else:
                cur_sents, cur_tokens = [], 0
            continue

        cur_sents.append(sent)
        cur_tokens += sent_tok
        i += 1

    if cur_sents:
        chunk_text = " ".join(cur_sents)
        if count_tokens_fn(chunk_text) >= min_tokens:
            chunks.append(chunk_text)

    return chunks


# ── Strategy 2: Blank-line block merging ───────────────────────────────
# Best for: student notes (topic boundaries at blank lines)
# Fallback: if a block > chunk_size, sub-chunk via sentence sliding window

_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def chunk_by_structure(
    text: str,
    count_tokens_fn: Callable[[str], int],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> List[str]:
    """Split notes at blank lines, greedily merge short adjacent blocks."""
    blocks = _BLANK_LINE_RE.split(text)
    chunks: List[str] = []
    buf_text, buf_tokens = "", 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block_tok = count_tokens_fn(block)

        if block_tok > chunk_size:
            if buf_text and buf_tokens >= min_tokens:
                chunks.append(buf_text.strip())
                buf_text, buf_tokens = "", 0
            chunks.extend(
                chunk_by_sentences(block, count_tokens_fn, chunk_size, min_tokens=min_tokens)
            )
            continue

        if buf_tokens + block_tok > chunk_size and buf_text:
            if buf_tokens >= min_tokens:
                chunks.append(buf_text.strip())
            buf_text, buf_tokens = block, block_tok
        else:
            buf_text = (buf_text + "\n\n" + block).strip()
            buf_tokens += block_tok

    if buf_text and buf_tokens >= min_tokens:
        chunks.append(buf_text.strip())

    return chunks


# ── Strategy 3: Per-slide ─────────────────────────────────────────────
# Best for: PPTX slides (each slide = self-contained semantic unit)

def chunk_by_slides(
    slides: List[Dict],
    count_tokens_fn: Callable[[str], int],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> List[str]:
    chunks: List[str] = []
    buf_text, buf_tokens = "", 0

    for slide in slides:
        slide_text = slide["text"]
        slide_tok  = count_tokens_fn(slide_text)

        if slide_tok > chunk_size:
            if buf_text and buf_tokens >= min_tokens:
                chunks.append(buf_text.strip())
            chunks.extend(
                chunk_by_sentences(slide_text, count_tokens_fn, chunk_size, min_tokens=min_tokens)
            )
            buf_text, buf_tokens = "", 0
            continue

        if buf_tokens + slide_tok > chunk_size and buf_text:
            if buf_tokens >= min_tokens:
                chunks.append(buf_text.strip())
            buf_text, buf_tokens = slide_text, slide_tok
        else:
            buf_text   = (buf_text + "\n\n" + slide_text).strip()
            buf_tokens += slide_tok

    if buf_text and buf_tokens >= min_tokens:
        chunks.append(buf_text.strip())

    return chunks


# ── Apply chunking ────────────────────────────────────────────────────

def apply_chunking(documents: List[Dict], count_tokens_fn: Callable) -> List[Dict]:
    """Chunk all documents using type-appropriate strategies."""
    strategy_map = {
        "transcript": ("sentence_sliding_window",
                       lambda doc: chunk_by_sentences(doc["text"], count_tokens_fn)),
        "notes":      ("blank_line_merge",
                       lambda doc: chunk_by_structure(doc["text"], count_tokens_fn)),
        "slides":     ("per_slide",
                       lambda doc: chunk_by_slides(doc["slides"], count_tokens_fn)),
    }

    all_chunks: List[Dict] = []
    seen_stems: Dict[str, int] = {}

    for doc in documents:
        strategy_name, chunker_fn = strategy_map[doc["doc_type"]]
        raw_chunks = chunker_fn(doc)

        stem = Path(doc["source"]).stem[:20]
        if stem in seen_stems:
            seen_stems[stem] += 1
            stem = f"{stem}_{seen_stems[stem]}"
        else:
            seen_stems[stem] = 0

        for idx, chunk_text in enumerate(raw_chunks):
            all_chunks.append({
                "chunk_id":    f"{stem}__c{idx:03d}",
                "source":      doc["source"],
                "doc_type":    doc["doc_type"],
                "chunk_index": idx,
                "strategy":    strategy_name,
                "text":        chunk_text,
                "token_count": count_tokens_fn(chunk_text),
            })

        print(f"  {doc['source'][:50]:<50}  {doc['doc_type']:12}  "
              f"{len(raw_chunks):>4} chunks  {strategy_name}")

    return all_chunks


# ── Embedding ─────────────────────────────────────────────────────────

def embed_chunks(all_chunks: List[Dict], model_name: str = EMBEDDING_MODEL):
    """Embed all chunk texts. Returns (model, embeddings_array)."""
    embed_model = SentenceTransformer(model_name)
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = embed_model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embed_model, embeddings


# ── Retrieval ─────────────────────────────────────────────────────────

def retrieve(query, embed_model, embeddings, all_chunks, top_k=3):
    """Return top-k chunks most similar to the query."""
    q_vec  = embed_model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(q_vec, embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return pd.DataFrame([{
        "rank":     r + 1,
        "score":    round(float(scores[i]), 4),
        "doc_type": all_chunks[i]["doc_type"],
        "source":   Path(all_chunks[i]["source"]).name[:38],
        "preview":  all_chunks[i]["text"][:100].replace("\n", " ").encode("ascii", "replace").decode() + " ...",
    } for r, i in enumerate(top_idx)])


# ── Topic concatenation ──────────────────────────────────────────────

def concatenate_by_topic(
    topic_queries: Dict[str, str],
    embed_model,
    embeddings: np.ndarray,
    all_chunks: List[Dict],
    output_dir,
) -> Dict:
    """
    Assign every chunk to its best-matching topic via cosine similarity,
    then write one combined .txt per topic to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_names  = list(topic_queries.keys())
    query_texts  = [topic_queries[k] for k in topic_names]
    q_vecs       = embed_model.encode(query_texts, normalize_embeddings=True)
    scores       = cosine_similarity(q_vecs, embeddings)   # (n_topics, n_chunks)
    assignments  = np.argmax(scores, axis=0)

    saved = {}
    for t_idx, name in enumerate(topic_names):
        chunk_indices = sorted(np.where(assignments == t_idx)[0])
        text = "\n\n".join(all_chunks[i]["text"] for i in chunk_indices)

        out_path = output_dir / f"{name}_ori.txt"
        out_path.write_text(text, encoding="utf-8")
        saved[name] = {"path": str(out_path), "n_chunks": len(chunk_indices), "chars": len(text)}
        print(f"  {out_path.name}: {len(chunk_indices)} chunks, {len(text):,} chars")

    return saved


# ── Stats & plotting ─────────────────────────────────────────────────

def print_stats(all_chunks: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_chunks)
    stats = (
        df.groupby("doc_type")["token_count"]
        .agg(
            n_chunks="count", mean="mean", median="median",
            min="min", max="max",
            over_200=lambda x: (x > 200).sum(),
            over_256=lambda x: (x > 256).sum(),
        )
        .round(1)
    )
    print(stats.to_string())
    over_256 = df[df["token_count"] > 256]
    if len(over_256):
        print(f"\n  {len(over_256)} chunk(s) exceed MiniLM 256-token limit (will be truncated):")
        print(over_256[["chunk_id", "doc_type", "token_count"]].to_string(index=False))
    else:
        print(f"\n  All {len(df)} chunks within MiniLM 256-token limit.")
    return df


def plot_distribution(df: pd.DataFrame, output_dir, chunk_size: int = CHUNK_SIZE_TOKENS):
    palette = {"transcript": "#55a868", "notes": "#c44e52", "slides": "#4c72b0"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    df["token_count"].hist(bins=30, ax=axes[0], color="#4C72B0", edgecolor="white", linewidth=0.5)
    axes[0].axvline(256, color="crimson", linestyle="--", linewidth=1.8, label="MiniLM limit (256)")
    axes[0].axvline(chunk_size, color="darkorange", linestyle="--", linewidth=1.8,
                    label=f"Target ({chunk_size})")
    axes[0].set_title("Token Count - All Chunks", fontsize=12)
    axes[0].set_xlabel("Tokens")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    for dtype, grp in df.groupby("doc_type"):
        grp["token_count"].hist(bins=20, ax=axes[1], alpha=0.65,
                                color=palette.get(dtype, "gray"), label=dtype)
    axes[1].axvline(256, color="crimson", linestyle="--", linewidth=1.8, label="MiniLM (256)")
    axes[1].set_title("Token Count by Document Type", fontsize=12)
    axes[1].set_xlabel("Tokens")
    axes[1].legend()

    plt.tight_layout()
    out_path = Path(output_dir) / "chunk_distribution.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ── Save outputs ──────────────────────────────────────────────────────

def save_outputs(all_chunks: List[Dict], embeddings: np.ndarray,
                 df: pd.DataFrame, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    chunks_path = output_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    embed_path = output_dir / "embeddings.npy"
    np.save(embed_path, embeddings)

    csv_path = output_dir / "chunk_metadata.csv"
    df.to_csv(csv_path, index=False)

    print(f"  chunks.json        : {len(all_chunks)} chunks")
    print(f"  embeddings.npy     : {embeddings.shape}")
    print(f"  chunk_metadata.csv : saved")
