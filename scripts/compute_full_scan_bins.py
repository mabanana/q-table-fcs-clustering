import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import yaml
from tqdm import tqdm
from src.fcs_loader import FCSLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main():
    config_path = Path("config.yaml")
    config = yaml.safe_load(config_path.read_text())

    markers = config["markers"]["use_for_clustering"]

    meta = pd.read_csv("data/AML/AML.csv")
    meta = meta[["FCSFileName", "Label"]].dropna()
    meta["FCSFileName"] = meta["FCSFileName"].astype(str).str.zfill(4)
    meta["filename"] = meta["FCSFileName"] + ".FCS"

    loader = FCSLoader(markers=markers)

    summaries = []
    missing_files = 0
    for fname in tqdm(meta["filename"], desc="Scanning FCS files"):
        file_path = Path("data/AML/FCS") / fname
        if not file_path.exists():
            missing_files += 1
            continue
        data, _ = loader.load_fcs_file(str(file_path))
        summary = data.median().to_dict()
        summary["filename"] = fname
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    merged = summary_df.merge(meta[["filename", "Label"]], on="filename", how="inner")

    print(f"Loaded summaries: {len(summary_df)} files")
    print(f"Missing files: {missing_files}")
    print(f"Labeled summaries: {len(merged)} files")

    if merged.empty:
        raise ValueError("No labeled summaries available for bin estimation")

    X = merged[markers].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    labels = merged["Label"].astype(str)
    if labels.str.contains("aml", case=False, na=False).any():
        y = labels.str.contains("aml", case=False, na=False).astype(int)
    else:
        y = pd.to_numeric(labels, errors="coerce")
        y = (y > 0).astype(int)

    quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    suggested_bins = {}
    for col in markers:
        q = X[col].quantile(quantiles).tolist()
        edges = [q[0], q[1], q[2], q[3], q[4]]
        if len(q) >= 6:
            edges.append(q[5])
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-6
        suggested_bins[col] = edges

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    bins_path = output_dir / "feature_bins_full_scan.yaml"
    bins_path.write_text(yaml.safe_dump(suggested_bins, sort_keys=False))

    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        coef = pd.Series(model.coef_[0], index=markers).abs().sort_values(ascending=False)
        coef_path = output_dir / "feature_importance_full_scan.csv"
        coef.to_csv(coef_path, header=["abs_coefficient"])
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        print("\nFeature importance (abs coef):")
        print(coef)
        print(f"\nTraining AUC (reference only): {auc:.3f}")

        top_features = coef.index[:3].tolist()
    except Exception as e:
        print(f"Logistic regression failed: {e}")
        top_features = config["discretization"].get("state_features", markers[:3])

    # Update config with full-scan bins and top features
    config["discretization"]["feature_bins"] = suggested_bins
    config["discretization"]["state_features"] = top_features

    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    print("\nSuggested quintile bins per marker (full scan):")
    for k, v in suggested_bins.items():
        print(f"  {k}: {v}")

    print(f"\nUpdated config state_features: {top_features}")
    print(f"Saved bins to: {bins_path}")
    print(f"Saved feature importance to: {coef_path if 'coef_path' in locals() else 'N/A'}")


if __name__ == "__main__":
    main()
