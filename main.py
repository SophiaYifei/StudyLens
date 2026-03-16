"""
main.py - StudyLens pipeline entry point.
Loads lecture materials, then for each index i combines dl_s{i}.pptx + dl_t{i}_cleaned.txt + dl_n{i}.txt
into dl_s{i}_ori.txt (and same for ml_*), producing 10 files in data/processed.
"""

import warnings

try:
    from StudyLens.scripts.naive import process_all_ppts
except ImportError:
    from scripts.naive import process_all_ppts
warnings.filterwarnings("ignore")

from pathlib import Path

from scripts.make_dataset import denoise_all_transcripts, load_all_documents
from scripts.build_features import write_per_slide_set_ori_files

# ── Paths ───────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent
DATA_DIR      = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # 1. Load documents from data/raw
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

    # 2. Denoise transcripts and process PPTs
    print("Denoising transcripts ...")
    denoise_all_transcripts(DATA_DIR)
    print("Processing PPTs ...")
    process_all_ppts(DATA_DIR)

    # 3. Write 10 _ori.txt files: each = slides + transcript + notes for that index
    print("\nWriting per-slide-set _ori.txt -> data/processed/")
    write_per_slide_set_ori_files(documents, PROCESSED_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
