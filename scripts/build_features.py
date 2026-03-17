"""
scripts/build_features.py
Fetch content from loaded raw data (PPT, notes, transcripts), create one combined
source file per topic in data/processed
Raw files are loaded by scripts.make_dataset.load_all_documents(data/raw); this
module consumes that document list.

AI Attribution: Code co-authored with Claude (Anthropic, https://claude.ai)
for structural design, debugging, and documentation.
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


# ── Per-slide-set _ori.txt (slides + transcript + notes per index) ────

_SLIDES_RE   = re.compile(r"^(\w+)_s(\d+)\.pptx$", re.IGNORECASE)
_TRANSCRIPT_RE = re.compile(r"^(\w+)_t(\d+)_cleaned\.txt$", re.IGNORECASE)
_NOTES_RE    = re.compile(r"^(\w+)_n(\d+)\.txt$", re.IGNORECASE)

# Remove decorative/placeholder symbols (slide ‹#›, bullet ▲, etc.) so output is plain text only
_SYMBOL_PLACEHOLDER_RE = re.compile(r"‹#›|[\u25B2\u25BC\u25C6\u2605\u2666\u2022]\s*")
def _strip_symbols(text: str) -> str:
    if not text:
        return text
    t = _SYMBOL_PLACEHOLDER_RE.sub("", text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return "\n".join(lines)


def write_per_slide_set_ori_files(documents: List[Dict], output_dir) -> Dict[str, Path]:
    """
    For each (prefix, num): combine dl_s{i}.pptx + dl_t{i}_cleaned.txt + dl_n{i}.txt
    into one file {prefix}_s{num}_ori.txt. Same for ml_*. Produces 10 files (e.g. dl_s1..dl_s5, ml_s1..ml_s5).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # (prefix, num) -> {"slides": text, "transcript": text, "notes": text}
    groups: Dict[tuple, Dict[str, str]] = {}

    for doc in documents:
        name = doc.get("source", "")
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        m = _SLIDES_RE.match(name)
        if m:
            key = (m.group(1).lower(), int(m.group(2)))
            groups.setdefault(key, {})["slides"] = text
            continue
        m = _TRANSCRIPT_RE.match(name)
        if m:
            key = (m.group(1).lower(), int(m.group(2)))
            groups.setdefault(key, {})["transcript"] = text
            continue
        m = _NOTES_RE.match(name)
        if m:
            key = (m.group(1).lower(), int(m.group(2)))
            groups.setdefault(key, {})["notes"] = text

    saved = {}
    for (prefix, num) in sorted(groups.keys()):
        g = groups[(prefix, num)]
        if "slides" not in g or "transcript" not in g or "notes" not in g:
            missing = [k for k in ("slides", "transcript", "notes") if k not in g]
            print(f"  Skip {prefix}_s{num}: missing {missing}")
            continue
        # Only content from data/raw; no extra labels; strip decorative symbols (‹#›, ▲, etc.)
        parts = [_strip_symbols(g["slides"]), _strip_symbols(g["transcript"]), _strip_symbols(g["notes"])]
        combined = "\n\n".join(p for p in parts if p)
        out_name = f"{prefix}_s{num}_ori.txt"
        out_path = output_dir / out_name
        out_path.write_text(combined, encoding="utf-8")
        saved[out_name] = out_path
        print(f"  {out_name}: {len(combined):,} chars (slides + transcript + notes)")
    return saved


# ── Topic concatenation ──────────────────────────────────────────────

def concatenate_by_topic(
    topic_queries: Dict[str, str],
    embed_model,
    embeddings: np.ndarray,
    all_chunks: List[Dict],
    output_dir,
    min_similarity: float = 0.35,
) -> Dict:
    """
    Assign each chunk to its best-matching topic via cosine similarity,
    but only if the similarity exceeds *min_similarity*.
    Chunks below the threshold are dropped as irrelevant noise.
    Writes one combined .txt per topic to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_names  = list(topic_queries.keys())
    query_texts  = [topic_queries[k] for k in topic_names]
    q_vecs       = embed_model.encode(query_texts, normalize_embeddings=True)
    scores       = cosine_similarity(q_vecs, embeddings)   # (n_topics, n_chunks)

    # Best topic per chunk and its similarity score
    best_topic = np.argmax(scores, axis=0)       # (n_chunks,)
    best_score = np.max(scores, axis=0)           # (n_chunks,)

    # Filter: only keep chunks whose best score exceeds threshold
    above_thresh = best_score >= min_similarity

    n_total   = len(all_chunks)
    n_kept    = int(above_thresh.sum())
    n_dropped = n_total - n_kept

    print(f"  Relevance threshold : {min_similarity}")
    print(f"  Chunks kept/total   : {n_kept}/{n_total}  ({n_dropped} dropped)")
    print(f"  Similarity stats    : min={best_score.min():.3f}  "
          f"mean={best_score.mean():.3f}  median={np.median(best_score):.3f}  "
          f"max={best_score.max():.3f}")

    saved = {}
    for t_idx, name in enumerate(topic_names):
        # Chunks assigned to this topic AND above threshold
        mask = (best_topic == t_idx) & above_thresh
        chunk_indices = sorted(np.where(mask)[0])

        # Stats: how many assigned to this topic were dropped
        assigned_total   = int((best_topic == t_idx).sum())
        assigned_dropped = assigned_total - len(chunk_indices)

        text = "\n\n".join(all_chunks[i]["text"] for i in chunk_indices)

        out_path = output_dir / f"{name}_ori.txt"
        out_path.write_text(text, encoding="utf-8")
        saved[name] = {
            "path": str(out_path),
            "n_chunks": len(chunk_indices),
            "n_dropped": assigned_dropped,
            "chars": len(text),
        }
        print(f"  {out_path.name}: {len(chunk_indices)} chunks "
              f"({assigned_dropped} dropped below {min_similarity}), "
              f"{len(text):,} chars")

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


