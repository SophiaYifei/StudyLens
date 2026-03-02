# ============================================
# scripts/model.py
# ============================================
from transformers import pipeline, AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk
import math
from pathlib import Path

nltk.download('punkt_tab')

# --- Base Class ---
class BaseSummarizer:
    """All summarizers must implement .summarize(text) -> str"""

    def summarize(self, text):
        raise NotImplementedError("Subclasses must implement summarize()")

# --- BART Summarizer ---
class BARTSummarizer(BaseSummarizer):

#     def __init__(self):
#         self.model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"
#         self.pipe = pipeline("summarization", model=self.model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.max_input_tokens = 16384

    # --- BART Summarizer ---
    def __init__(self):
        # Load the HuggingFace pipeline once
        # model = "facebook/bart-large-cnn"
        # max_input_tokens = 1024
        # store these as self.xxx
        self.model_name = "facebook/bart-large-cnn"
        self.pipe = pipeline("summarization", model=self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_input_tokens = 1024

    def _count_tokens(self, text):
        # Use self.tokenizer to count tokens in text
        # Return integer
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_into_chunks(self, text):
        # Step 1: Split text into sentences (use nltk sent_tokenize)
        sentences = sent_tokenize(text)

        # Step 2: Count total tokens (I changed it to Step 3)

        # Step 3: Count tokens for each sentence (so we don't recount later)
        sent_token_counts = [self._count_tokens(s) for s in sentences]
        total_tokens = sum(sent_token_counts)

        # Step 4: Calculate minimum number of chunks, then balanced target
        num_chunks = math.ceil(total_tokens / self.max_input_tokens)
        target = total_tokens / num_chunks

        # Step 5: Fill chunks using prefix sum to find nearest sentence boundary
        #         to each ideal cut point
        chunks = []
        current_chunk = []
        current_tokens = 0
        # Which chunk we're filling (0-indexed)
        chunk_index = 0

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_tokens += sent_token_counts[i]

            # The ideal total tokens consumed by the time we finish this chunk
            ideal_cutoff = target * (chunk_index + 1)

            # Should we cut here? Only if:
            # 1. We've reached or passed the ideal cutoff
            # 2. This isn't the last chunk (last chunk just takes everything remaining)
            if current_tokens >= ideal_cutoff and chunk_index < num_chunks - 1:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                chunk_index += 1

        # Last chunk gets whatever is left
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Return list of chunk strings
        return chunks


    def _summarize_single(self, text: str, max_length=150, min_length=40) -> str:
        # Call self.pipe(text, max_length=..., min_length=..., do_sample=False)
        # Return the summary_text string

        # Safety checks
        text_tokens = self._count_tokens(text)
        if text_tokens < min_length:
            # Adjust min_length downwards for very short inputs
            min_length = max(10, text_tokens // 2)
        if max_length <= min_length:
            # Ensure max_length is always greater than min_length
            max_length = min_length + 20

        result = self.pipe(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        return result[0]['summary_text']

    def summarize(self, text: str) -> str:
        # This is the main method called from outside

        # Step 1: Count tokens of input text
        total_tokens = self._count_tokens(text)

        # Step 2: If tokens <= 1024, just call _summarize_single directly
        if total_tokens <= self.max_input_tokens:
            print(f"Text fits in one pass ({total_tokens} tokens), summarizing directly.")
            max_len = min(total_tokens // 2, 512)
            return self._summarize_single(text, max_length=max_len, min_length=40)
        
        # Step 3: If tokens > 1024 (hierarchical summarization):
        print(f"Text too long ({total_tokens} tokens), performing hierarchical summarization.")

        #   a. chunks = self._split_into_chunks(text)
        chunks = self._split_into_chunks(text)
        print(f"Split text into {len(chunks)} chunks.")

        #   b. chunk_summaries = []
        #      for each chunk:
        #          summary = self._summarize_single(chunk)
        #          chunk_summaries.append(summary)
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = self._count_tokens(chunk)
            print(f"Summarizing chunk {i+1}/{len(chunks)} ({chunk_tokens} tokens)...")
            max_len = min(chunk_tokens // 2, 400)
            summary = self._summarize_single(chunk, max_length=max_len, min_length=40)
            chunk_summaries.append(summary)

        #   c. combined = join all chunk_summaries together
        combined = " ".join(chunk_summaries)
        combined_tokens = self._count_tokens(combined)
        print(f"Combined chunk summaries into one text ({combined_tokens} tokens).")

        #   d. final_summary = self._summarize_single(combined)  ← second pass
        if combined_tokens > self.max_input_tokens:
            print(f"Combined summary still too long ({combined_tokens} tokens), summarizing again with truncation.")
            return self.summarize(combined)
        else:
            print(f"Final pass: generating coherent summary...")
            max_len = min(combined_tokens, 600)
            return self._summarize_single(combined, max_length=max_len, min_length=40)

        # #   d. final_summary
        # if combined_tokens > self.max_input_tokens:
        #     print(f"Combined summary still too long ({combined_tokens} tokens), summarizing again.")
        #     return self.summarize(combined)
        # else:
        #     # If combined is short enough, just return it directly
        #     # The chunk summaries together already form a good summary
        #     print(f"    Combined summary: {combined_tokens} tokens, {len(combined.split())} words")
        #     return combined
# --- Helper function to process all files ---
def process_all_topics(input_dir, output_dir, model):
    """
    Loop through all _ori.txt files in input_dir,
    summarize each, save as _sum.txt in output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Step 1: Find all files ending with _ori.txt in input_dir
    ori_files = sorted(input_path.glob("*_ori.txt"))

    if not ori_files:
        print(f"No files ending with '_ori.txt' found in {input_dir}.")
        return {}
    print(f"Found {len(ori_files)} files to process.")

    results = {}

    # Step 2: For each file:
    for file in ori_files:
    #     a. Read the text
        text = file.read_text(encoding="utf-8")
        print(f"Input length: {len(text.split())} words")

    #     b. summary = model.summarize(text)
        summary = model.summarize(text)
        print(f"Summary length: {len(summary.split())} words")

    #     c. Create output filename: replace _ori with _sum
        output_filename = file.name.replace("_ori.txt", "_sum.txt")
        output_file = output_path / output_filename

    #     d. Write summary to output_dir / output_filename
        output_file.write_text(summary, encoding="utf-8")
        print(f"Saved summary to {output_file}")

    #     e. Print progress: which file, token count, summary length
        results[file.name] = {
            "input_file": str(file),
            "output_file": str(output_file),
            "input_words": len(text.split()),
            "summary_words": len(summary.split()),
        }

    print(f"Summarized {len(results)} files.")
        
    return results

# --- Test block ---
if __name__ == "__main__":
    # Quick test to verify everything works
    # model = BARTSummarizer()
    # process_all_topics("data/processed", "data/outputs", model)
    model = BARTSummarizer()

    results = process_all_topics("data/processed", "data/outputs", model)

    for filename, info in results.items():
        print(f"{filename}: {info['input_words']} words -> {info['summary_words']} words")