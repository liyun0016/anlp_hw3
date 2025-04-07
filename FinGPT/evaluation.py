import os
import pandas as pd
import re

def extract_prediction_fuzzy(text):
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    if any(word in text_lower for word in ['increase', 'gain', 'rise', 'appreciate']):
        direction = 'U'
    elif any(word in text_lower for word in ['decrease', 'drop', 'decline', 'fall', 'down']):
        direction = 'D'
    else:
        return None
    percent_match = re.search(r'(\d+(\.\d+)?)-(\d+(\.\d+)?)%', text)
    if percent_match:
        lower_bound = float(percent_match.group(1))
    else:
        single_match = re.search(r'(\d+(\.\d+)?)%', text)
        if single_match:
            lower_bound = float(single_match.group(1))
        else:
            return direction + "1"
    return direction + ("5+" if lower_bound >= 5 else str(int(lower_bound)))

def match_direction(true_label, pred_label):
    if pd.isna(true_label) or pd.isna(pred_label):
        return False
    return true_label[0] == pred_label[0]

def evaluate_all_symbols(data_dir, start_date, end_date):
    DOW_30 = [
        "AXP", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON",
        "IBM", "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE",
        "PG", "TRV", "UNH", "CRM", "VZ", "V", "WBA", "WMT", "DIS", "DOW"
    ]
    results = []
    for symbol in DOW_30:
        gt_path = os.path.join(data_dir, f"{symbol}_{start_date}_{end_date}.csv")
        pred_path = os.path.join(data_dir, f"{symbol}_{start_date}_{end_date}_mistral.csv")
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            continue
        try:
            gt_df = pd.read_csv(gt_path)
            pred_df = pd.read_csv(pred_path)
            pred_df["pred_label"] = pred_df["answer"].apply(extract_prediction_fuzzy)
            min_len = min(len(gt_df), len(pred_df))
            gt_df = gt_df.iloc[:min_len].reset_index(drop=True)
            pred_df = pred_df.iloc[:min_len].reset_index(drop=True)
            df = pd.DataFrame({
                "true_label": gt_df["Bin Label"],
                "pred_label": pred_df["pred_label"]
            })
            df["correct_full"] = df["true_label"] == df["pred_label"]
            df["correct_direction"] = df.apply(lambda r: match_direction(r["true_label"], r["pred_label"]), axis=1)
            results.append({
                "symbol": symbol,
                "samples": len(df),
                "directional_accuracy": df["correct_direction"].mean(),
                "full_accuracy": df["correct_full"].mean(),
                "missing_predictions": df["pred_label"].isna().sum()
            })
        except Exception as e:
            results.append({
                "symbol": symbol,
                "error": str(e)
            })

    df_results = pd.DataFrame(results)
    df_clean = df_results[df_results["directional_accuracy"].notna()].copy()
    mean_row = {
        "symbol": "Average",
        "samples": df_clean["samples"].mean(),
        "directional_accuracy": df_clean["directional_accuracy"].mean(),
        "full_accuracy": df_clean["full_accuracy"].mean(),
        "missing_predictions": df_clean["missing_predictions"].mean()
    }
    df_results = pd.concat([df_clean, pd.DataFrame([mean_row])], ignore_index=True)

    output_csv = os.path.join(data_dir, "evaluation_summary.csv")
    df_results.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")

    return df_results

# Run
data_dir = "data/DOW-30_2023-5-31_2023-12-31"
start_date = "2023-5-31"
end_date = "2023-12-31"
evaluate_all_symbols(data_dir, start_date, end_date)
