from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "artifacts").mkdir(exist_ok=True)

    df = pd.read_csv(root / "data" / "lectures_dataset.csv")
    threshold = float(df["median_engagement"].median())
    df["engagement_label"] = (df["median_engagement"] >= threshold).astype(int)

    y = df["engagement_label"]
    X = df.drop(columns=["median_engagement", "engagement_label"])
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (root / "results" / "metrics_video_baseline.json").write_text(
        json.dumps(metrics, indent=2)
    )
    joblib.dump(model, root / "results" / "artifacts" / "video_logreg.joblib")
    print(metrics)


if __name__ == "__main__":
    main()
