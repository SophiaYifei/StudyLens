import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def main() -> None:
    root = Path("data") / "outputs" / "eval"
    csv_path = root / "evaluation_results_all.csv"
    print(f"Reading {csv_path} ...")

    df = pd.read_csv(csv_path)

    baselines = ["naive_first5", "tfidf", "bart", "led-arxiv"]
    base_df = df[df["model"].isin(baselines)].copy()

    if base_df.empty:
        print("No baseline rows found for", baselines)
        return

    # ROUGE-L F1 by model
    rouge = base_df.groupby("model")["rouge_l_f1"].mean().reindex(baselines)
    plt.figure(figsize=(8, 4))
    rouge.plot(kind="bar")
    plt.ylabel("ROUGE-L F1")
    plt.title("ROUGE-L F1 (baselines only)")
    plt.tight_layout()
    out1 = root / "rouge_l_f1_baselines_from_csv.png"
    plt.savefig(out1)
    plt.close()
    print("Saved", out1)

    # BERTScore F1 by model
    bert = base_df.groupby("model")["bertscore_f1"].mean().reindex(baselines)
    plt.figure(figsize=(8, 4))
    bert.plot(kind="bar")
    plt.ylabel("BERTScore F1")
    plt.title("BERTScore F1 (baselines only)")
    plt.tight_layout()
    out2 = root / "bertscore_f1_baselines_from_csv.png"
    plt.savefig(out2)
    plt.close()
    print("Saved", out2)


if __name__ == "__main__":
    main()

