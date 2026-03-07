from pathlib import Path
from pptx import Presentation
from zipfile import BadZipFile
import re

target_directory = Path("StudyLens/data/output/naive")


def get_first_sentence_from_first_5_slides(filepath):
    prs = Presentation(filepath)
    collected_sentences = []

    for i, slide in enumerate(prs.slides):
        if i >= 5:
            break

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text = shape.text.strip()

                sentences = re.split(r'(?<=[.!?])\s+', text)

                for sentence in sentences:
                    clean = sentence.strip()
                    if clean:
                        collected_sentences.append(clean)
                        break  # first sentence from this slide

        # continue to next slide automatically

    return collected_sentences


def process_all_ppts(directory: Path):
    for ppt_file in directory.glob("*.pptx"):

        if ppt_file.name.startswith("~$"):
            continue

        try:
            sentences = get_first_sentence_from_first_5_slides(ppt_file)

            output_file = ppt_file.with_name(ppt_file.stem + "_first5.txt")
            output_file.write_text("\n".join(sentences) + "\n", encoding="utf-8")

            print(f"Created: {output_file}")

        except BadZipFile:
            print(f"Skipped (not valid pptx): {ppt_file}")
        except Exception as e:
            print(f"Error processing {ppt_file}: {e}")

if __name__ == "__main__":
    process_all_ppts(target_directory)