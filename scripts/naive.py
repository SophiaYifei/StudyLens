from pathlib import Path
from pptx import Presentation
from zipfile import BadZipFile
import re

target_directory = Path("StudyLens/data/raw")

def get_first_sentence_min_words(filepath, min_words=10):
    prs = Presentation(filepath)

    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text = shape.text.strip()

                # split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', text)

                for sentence in sentences:
                    clean = sentence.strip()
                    word_count = len(re.findall(r"\b\w+\b", clean))
                    if word_count >= min_words:
                        return clean

    return ""


def process_all_ppts(directory: Path):
    for ppt_file in directory.glob("*.pptx"):

        if ppt_file.name.startswith("~$"):
            continue

        try:
            sentence = get_first_sentence_min_words(ppt_file, min_words=10)

            output_file = ppt_file.with_name(ppt_file.stem + "_naive.txt")
            output_file.write_text(sentence + "\n", encoding="utf-8")

            print(f"Created: {output_file}")

        except BadZipFile:
            print(f"Skipped (not valid pptx): {ppt_file}")
        except Exception as e:
            print(f"Error processing {ppt_file}: {e}")


