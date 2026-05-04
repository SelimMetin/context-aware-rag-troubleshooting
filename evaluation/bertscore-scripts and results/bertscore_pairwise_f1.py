

import itertools
import pandas as pd
from bert_score import score as bertscore

INPUT_XLSX = "input_EN.xlsx"
OUTPUT_XLSX = "bert_score_results_EN_PAIRWISE_F1.xlsx"

MODEL_TYPE = "dbmdz/bert-base-turkish-cased"
LANG = "en"

SYSTEM_COLUMNS = [
    "Proposed system",
    "LLM-EB",
    "LLM-BBC",
    "Proposed system (w/o system-level context)",
    "Proposed system (w/o application-level context)",
]


def to_str_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def compute_bertscore_f1(cands: list[str], refs: list[str], model_type: str, lang: str):
    _, _, f1 = bertscore(
        cands=cands,
        refs=refs,
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=True,
        verbose=True,
    )
    return f1.tolist()


def main():
    df = pd.read_excel(INPUT_XLSX)

    missing = [c for c in SYSTEM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Beklenen sütun(lar) bulunamadı: {missing}\nBulunan sütunlar: {list(df.columns)}")

    for col in SYSTEM_COLUMNS:
        df[col] = to_str_series(df[col])

    row_level_results = []
    summary_rows = []

    for sys_a, sys_b in itertools.combinations(SYSTEM_COLUMNS, 2):
        pair_df = df[[sys_a, sys_b]].copy()

        valid_mask = (pair_df[sys_a] != "") & (pair_df[sys_b] != "")
        valid_df = pair_df[valid_mask].copy()

        if valid_df.empty:
            summary_rows.append({
                "system_1": sys_a,
                "system_2": sys_b,
                "n_used": 0,
                "mean_F1": None,
                "std_F1": None,
                "cell_text": "N/A",
            })
            continue

        cands = valid_df[sys_a].tolist()
        refs = valid_df[sys_b].tolist()

        f1_scores = compute_bertscore_f1(cands, refs, MODEL_TYPE, LANG)

        valid_df = valid_df.reset_index().rename(columns={"index": "original_row"})
        valid_df["system_1"] = sys_a
        valid_df["system_2"] = sys_b
        valid_df["bertscore_F1"] = f1_scores

        row_level_results.append(valid_df[[
            "original_row", "system_1", "system_2", sys_a, sys_b, "bertscore_F1"
        ]].rename(columns={sys_a: "text_1", sys_b: "text_2"}))

        mean_f1 = valid_df["bertscore_F1"].mean()
        std_f1 = valid_df["bertscore_F1"].std(ddof=1)

        summary_rows.append({
            "system_1": sys_a,
            "system_2": sys_b,
            "n_used": len(valid_df),
            "mean_F1": mean_f1,
            "std_F1": std_f1,
            "cell_text": f"{mean_f1:.4f}±{std_f1:.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)

    matrix_df = pd.DataFrame("", index=SYSTEM_COLUMNS, columns=SYSTEM_COLUMNS)
    for col in SYSTEM_COLUMNS:
        matrix_df.loc[col, col] = "1.0000±0.0000"

    for _, row in summary_df.iterrows():
        matrix_df.loc[row["system_1"], row["system_2"]] = row["cell_text"]
        matrix_df.loc[row["system_2"], row["system_1"]] = row["cell_text"]

    mean_f1_df = pd.DataFrame(index=SYSTEM_COLUMNS, columns=SYSTEM_COLUMNS, dtype=float)
    std_f1_df = pd.DataFrame(index=SYSTEM_COLUMNS, columns=SYSTEM_COLUMNS, dtype=float)
    for col in SYSTEM_COLUMNS:
        mean_f1_df.loc[col, col] = 1.0
        std_f1_df.loc[col, col] = 0.0

    for _, row in summary_df.iterrows():
        s1, s2 = row["system_1"], row["system_2"]
        mean_f1_df.loc[s1, s2] = row["mean_F1"]
        mean_f1_df.loc[s2, s1] = row["mean_F1"]
        std_f1_df.loc[s1, s2] = row["std_F1"]
        std_f1_df.loc[s2, s1] = row["std_F1"]

    row_scores_df = pd.concat(row_level_results, ignore_index=True) if row_level_results else pd.DataFrame()

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        matrix_df.to_excel(writer, sheet_name="BERTScore_Table")
        summary_df.to_excel(writer, sheet_name="Summary_Long", index=False)
        mean_f1_df.to_excel(writer, sheet_name="Mean_F1")
        std_f1_df.to_excel(writer, sheet_name="Std_F1")
        row_scores_df.to_excel(writer, sheet_name="Row_Level_F1", index=False)

    print(f"✅ BERTScore pairwise F1-only analysis tamamlandı. Çıktı: {OUTPUT_XLSX}")
    print("\nPairwise mean±std F1 summary:")
    print(summary_df[["system_1", "system_2", "n_used", "cell_text"]].to_string(index=False))


if __name__ == "__main__":
    main()
