"""
main.py - StudyLens pipeline entry point.
Loads lecture materials, chunks, embeds, and produces per-topic text files
for downstream BART summarization.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from transformers import AutoTokenizer

from scripts.make_dataset import load_all_documents
from scripts.build_features import (
    make_token_counter,
    apply_chunking,
    embed_chunks,
    concatenate_by_topic,
    print_stats,
    plot_distribution,
    save_outputs,
    retrieve,
    TOKENIZER_MODEL,
    EMBEDDING_MODEL,
)

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent
DATA_DIR      = ROOT_DIR / "data" / "raw"
OUTPUT_DIR    = ROOT_DIR / "outputs"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ── Topic queries for concatenation ──────────────────────────────────────
# Keys become filenames: {key}_ori.txt  ->  data/processed/
TOPIC_QUERIES = {
    "dl_s6_neural_networks_nlp": (
        "Neural Networks for NLP: word embeddings word2vec GloVe "
        "RNN LSTM GRU sequence models"
    ),
    "dl_s7_attention_transformers": (
        "Attention & Transformers: self-attention keys queries values "
        "multi-head transformer BERT GPT"
    ),
}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load documents
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    print("Loading documents ...")
    documents = load_all_documents(DATA_DIR)

    if not documents:
        raise RuntimeError(f"No .txt or .pptx files found in {DATA_DIR}")

    for doc in documents:
        label = (doc["metadata"].get("num_slides")
                 and f"{doc['metadata']['num_slides']} slides") \
                or f"{doc['metadata']['chars']:,} chars"
        print(f"  {doc['doc_type']:12s}  {doc['source'][:55]}  ({label})")
    print(f"  {len(documents)} documents loaded.\n")

    # 2. Tokenizer
    print(f"Loading tokenizer: {TOKENIZER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    count_tokens = make_token_counter(tokenizer)

    # 3. Chunk
    print("\nChunking ...")
    all_chunks = apply_chunking(documents, count_tokens)
    print(f"\n  Total chunks: {len(all_chunks)}\n")

    # 4. Stats
    df = print_stats(all_chunks)
    plot_distribution(df, OUTPUT_DIR)

    # 5. Embed
    print(f"\nEmbedding {len(all_chunks)} chunks with {EMBEDDING_MODEL} ...")
    embed_model, embeddings = embed_chunks(all_chunks)
    print(f"  Embeddings: {embeddings.shape}\n")

    # 6. Retrieval sanity check
    print("Retrieval sanity check:")
    for q in TOPIC_QUERIES.values():
        print(f"\n  QUERY: {q}")
        print(retrieve(q, embed_model, embeddings, all_chunks).to_string(index=False))

    # 7. Concatenate per-topic text for BART summarization
    print(f"\nConcatenating text per topic -> data/processed/")
    concatenate_by_topic(TOPIC_QUERIES, embed_model, embeddings, all_chunks, PROCESSED_DIR)

    # 8. Save chunking/embedding outputs
    print(f"\nSaving outputs -> outputs/")
    save_outputs(all_chunks, embeddings, df, OUTPUT_DIR)

    print(f"\n  Documents loaded : {len(documents)}")
    print(f"  Total chunks     : {len(all_chunks)}")
    print(f"  Embedding shape  : {embeddings.shape}")
    print(f"  Topic files      : data/processed/")
    print(f"  Chunk outputs    : outputs/")
    print("Done.")


if __name__ == "__main__":
    main()
