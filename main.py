import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier
import time
from config import OPENML_DATASETS

results = []

def load_openml_data(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(dataset.default_target_attribute)

    y = LabelEncoder().fit_transform(y)  # TabPFN 需要数字标签
    return X, y

for did in OPENML_DATASETS:
    print(f"\n===== Running Dataset {did} =====")

    X, y = load_openml_data(did)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # ---------------- TabPFN ----------------
    tabpfn = TabPFNClassifier(N_ensemble_configurations=8)
    start = time.time()
    tabpfn.fit(X_train, y_train)
    pred = tabpfn.predict(X_test)
    tabpfn_time = time.time() - start
    tabpfn_acc = accuracy_score(y_test, pred)

    # ---------------- XGBoost ----------------
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="logloss")
    start = time.time()
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    xgb_time = time.time() - start
    xgb_acc = accuracy_score(y_test, pred)

    # ---------------- Save result ----------------
    results.append({
        "dataset": did,
        "tabpfn_acc": tabpfn_acc,
        "tabpfn_time": tabpfn_time,
        "xgb_acc": xgb_acc,
        "xgb_time": xgb_time,
    })

    print(f" TabPFN: acc={tabpfn_acc:.4f}, time={tabpfn_time:.2f}s")
    print(f" XGBoost: acc={xgb_acc:.4f}, time={xgb_time:.2f}s")

# ---------------- Save CSV ----------------
df = pd.DataFrame(results)
df.to_csv("results/results.csv", index=False)

print("\n🔥 ALL DONE — Results saved to results/results.csv")
print(df)

