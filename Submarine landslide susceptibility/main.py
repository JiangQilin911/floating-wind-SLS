# -*- coding: utf-8 -*-
"""
海底滑坡易发性评价
"""

import os
import argparse
import pandas as pd
import numpy as np
import warnings
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

METRIC_KEYS = ["AUC", "Accuracy", "F1", "Precision", "Recall"]


# ==================== 评估函数 ====================
def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_type="sklearn",
    threshold=None,
):
    if model_type == "pytorch":
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            train_proba = (
                model(torch.tensor(X_train, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
                .flatten()
            )
            test_proba = (
                model(torch.tensor(X_test, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
                .flatten()
            )
    else:
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

    train_proba = np.clip(train_proba, 0, 1)
    test_proba = np.clip(test_proba, 0, 1)

    if threshold is None:
        fpr_tr, tpr_tr, th_tr = roc_curve(y_train, train_proba, pos_label=1)
        threshold = (
            float(th_tr[np.argmax(tpr_tr - fpr_tr)]) if len(th_tr) > 0 else 0.5
        )

    train_pred = (train_proba >= threshold).astype(int)
    test_pred = (test_proba >= threshold).astype(int)

    train_scores = {
        "AUC": roc_auc_score(y_train, train_proba),
        "Accuracy": accuracy_score(y_train, train_pred),
        "F1": f1_score(y_train, train_pred),
        "Precision": precision_score(y_train, train_pred),
        "Recall": recall_score(y_train, train_pred),
    }

    test_scores = {
        "AUC": roc_auc_score(y_test, test_proba),
        "Accuracy": accuracy_score(y_test, test_pred),
        "F1": f1_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred),
        "Recall": recall_score(y_test, test_pred),
    }

    return train_scores, test_scores, threshold, test_proba


def format_metrics(scores):
    return (
        f"AUC={scores['AUC']:.4f}  Acc={scores['Accuracy']:.4f}  "
        f"F1={scores['F1']:.4f}  Pre={scores['Precision']:.4f}  Rec={scores['Recall']:.4f}"
    )


# ==================== 特征相关性剔除 ====================
def remove_highly_correlated_features(X, y, feature_names, threshold=0.85):
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    target_corr = df.corr()["target"].drop("target").abs()
    corr_matrix = df[feature_names].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    drop_features = set()
    for col in upper_tri.columns:
        for row in upper_tri.index:
            val = upper_tri.loc[row, col]
            if not np.isnan(val) and val >= threshold:
                if col in drop_features or row in drop_features:
                    continue

                if target_corr[col] >= target_corr[row]:
                    drop_features.add(row)
                    print(f"    高相关 {col} - {row} (r={val:.3f})，保留 {col}，剔除 {row}")
                else:
                    drop_features.add(col)
                    print(f"    高相关 {col} - {row} (r={val:.3f})，保留 {row}，剔除 {col}")

    kept_features = [f for f in feature_names if f not in drop_features]
    print(f"    剔除 {len(drop_features)} 个特征，保留 {len(kept_features)} 个特征: {kept_features}")
    return df[kept_features].values, kept_features


# ==================== 网络定义 ====================
class SimpleMLP(nn.Module):
    def __init__(self, n_features, hidden_dims=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(1)


class SimpleCNN(nn.Module):
    def __init__(self, n_features, n_channels=16, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, n_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(
            n_channels, n_channels * 2, kernel_size=kernel_size, padding=1
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_channels * 2, 32)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.avgpool(x).squeeze(-1)
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        return torch.sigmoid(self.output(x)).squeeze(1)


class SimpleRNN(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size=32,
        num_layers=1,
        dropout=0.2,
        rnn_type="lstm",
    ):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_type = rnn_type.lower()

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                1,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                1,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.RNN(
                1,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )

        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out)).squeeze(1)


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        n_features,
        d_model=64,
        nhead=4,
        num_layers=3,
        dropout=0.1,
        use_cls=True,
    ):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_features, d_model))
        self.use_cls = use_cls

        if use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x) + self.pos_embedding

        if self.use_cls:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        x = self.transformer(x)
        out = x[:, 0, :] if self.use_cls else x.mean(dim=1)
        out = self.dropout(out)
        return torch.sigmoid(self.output_layer(out)).squeeze(1)


# ==================== 训练深度模型 ====================
def train_deep_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=80,
    batch_size=256,
    lr=1e-3,
    patience=15,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_auc = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for Xb, _ in val_loader:
                preds.append(model(Xb.to(device)).cpu().numpy())

        preds = np.concatenate(preds)

        try:
            val_auc = roc_auc_score(y_val, preds)
        except ValueError:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_auc


class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_type="mlp",
        n_features=9,
        hidden_dims=(64, 32),
        n_channels=16,
        kernel_size=3,
        hidden_size=32,
        num_layers=1,
        rnn_type="lstm",
        d_model=64,
        nhead=4,
        trans_layers=3,
        dropout=0.2,
        epochs=80,
        batch_size=256,
        lr=1e-3,
        patience=15,
        random_state=42,
    ):
        self.model_type = model_type
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.d_model = d_model
        self.nhead = nhead
        self.trans_layers = trans_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.random_state = random_state

    def _build(self):
        if self.model_type == "mlp":
            return SimpleMLP(
                self.n_features,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )

        if self.model_type == "cnn":
            return SimpleCNN(
                self.n_features,
                n_channels=self.n_channels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            )

        if self.model_type == "rnn":
            return SimpleRNN(
                self.n_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                rnn_type=self.rnn_type,
            )

        if self.model_type == "transformer":
            return SimpleTransformer(
                self.n_features,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.trans_layers,
                dropout=self.dropout,
            )

        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model_ = self._build()

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X,
                y,
                test_size=0.1,
                random_state=self.random_state,
                stratify=y,
            )
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X,
                y,
                test_size=0.1,
                random_state=self.random_state,
            )

        self.model_, _ = train_deep_model(
            self.model_,
            X_tr,
            y_tr,
            X_val,
            y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
        )

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        device = next(self.model_.parameters()).device
        self.model_.eval()

        with torch.no_grad():
            p = (
                self.model_(torch.tensor(X, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
                .flatten()
            )

        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ==================== SHAP 计算 ====================
def shap_feature_importance(
    best_model,
    best_model_name,
    X_train,
    X_test,
    y_test,
    feature_names,
    model_type="sklearn",
    max_background=100,
    max_explain=200,
):
    try:
        import shap
    except ImportError as e:
        print(f"    SHAP 未安装或导入失败: {e}")
        return None

    print(f"\n[SHAP 分析] 目标模型 = {best_model_name} ({model_type})")

    rng = np.random.RandomState(RANDOM_STATE)
    X_bg = X_train[
        rng.choice(len(X_train), min(max_background, len(X_train)), replace=False)
    ]
    X_ex = X_test[
        rng.choice(len(X_test), min(max_explain, len(X_test)), replace=False)
    ]

    shap_values = None

    try:
        if model_type == "pytorch":
            device = next(best_model.parameters()).device

            def predict_fn(X):
                best_model.eval()
                with torch.no_grad():
                    return (
                        best_model(torch.tensor(X, dtype=torch.float32).to(device))
                        .cpu()
                        .numpy()
                        .flatten()
                    )

            explainer = shap.KernelExplainer(predict_fn, X_bg)
            shap_values = explainer.shap_values(X_ex, nsamples=100, silent=True)

        else:
            try:
                explainer = shap.TreeExplainer(best_model)
                sv = explainer.shap_values(X_ex)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                shap_values = sv

            except Exception:
                try:
                    explainer = shap.Explainer(best_model.predict_proba, X_bg)
                    sv = explainer(X_ex).values
                    if sv.ndim == 3:
                        sv = sv[:, :, 1]
                    shap_values = sv

                except Exception:
                    explainer = shap.PermutationExplainer(
                        lambda X: best_model.predict_proba(X)[:, 1], X_bg
                    )
                    sv = explainer(X_ex).values
                    if sv.ndim == 3:
                        sv = sv[:, :, 1]
                    shap_values = sv

    except Exception as e:
        print(f"    SHAP 计算失败: {e}")
        return None

    if shap_values is None:
        print("    SHAP 值为空，跳过。")
        return None

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean_Abs_SHAP": np.round(mean_abs_shap, 4),
            }
        )
        .sort_values("Mean_Abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )

    importance_df.insert(0, "Rank", np.arange(1, len(importance_df) + 1))
    print(importance_df.to_string(index=False))

    return importance_df


# ==================== 主程序 ====================
def main(input_csv=None, output_dir=None, encoding="gbk"):
    parser = argparse.ArgumentParser(
        description="海底滑坡易发性评价 —— 11 模型统一 5 折 CV 网格搜索"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=input_csv if input_csv else "data/samples.csv",
        help="输入 CSV 路径，默认 data/samples.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir if output_dir else "outputs",
        help="输出目录，默认 outputs",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=encoding,
        help="CSV 编码，默认 gbk；如 UTF-8 可传 utf-8",
    )

    args = parser.parse_args()

    input_csv = args.input_csv
    save_dir = args.output_dir
    encoding = args.encoding

    feature_cols = [
        "水深",
        "坡度",
        "坡向",
        "曲率",
        "距断层距离",
        "地形起伏度",
        "地面粗糙度",
        "沉积物类型",
        "PGA",
    ]
    target_col = "label"

    print("=" * 78)
    print("海底滑坡易发性评价 —— 11 模型统一 5 折 CV 网格搜索")
    print("模型选择依据：5 折 CV 平均 AUC（不使用测试集选择模型）")
    print("最优阈值来源：训练集 ROC 的 Youden 指数最优点")
    print("输出内容：仅 3 个 CSV 文件，不生成任何图件")
    print("=" * 78)
    print(f"输入 CSV: {input_csv}")
    print(f"输出目录: {save_dir}")
    print(f"编码: {encoding}")

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(input_csv, encoding=encoding)
    first_three_cols = df.iloc[:, :3].copy().reset_index(drop=True)

    available = [f for f in feature_cols if f in df.columns]
    X = df[available].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ---------- 特征相关性剔除 ----------
    print("\n[特征工程] 相关性分析（阈值 = 0.85）...")
    X_train, kept_features = remove_highly_correlated_features(
        X_train, y_train, available, threshold=0.85
    )

    X_test = pd.DataFrame(X_test, columns=available)[kept_features].values
    X_total = pd.DataFrame(X, columns=available)[kept_features].values
    n_features = len(kept_features)

    print(f"    最终使用 {n_features} 个特征: {kept_features}")

    # ---------- 标准化 ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_total_scaled = scaler.transform(X_total)

    # ---------- 所有模型配置 ----------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    sklearn_models = {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=2000,
                class_weight="balanced",
            ),
            "params": {"C": [0.1, 1.0, 10.0]},
            "n_jobs": -1,
        },
        "SVM": {
            "estimator": SVC(
                probability=True,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            "params": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
            "n_jobs": -1,
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 8, None],
            },
            "n_jobs": -1,
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
            "n_jobs": -1,
        },
        "XGBoost": {
            "estimator": xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
            },
            "n_jobs": -1,
        },
        "LightGBM": {
            "estimator": lgb.LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1,
                class_weight="balanced",
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
            },
            "n_jobs": -1,
        },
        "CatBoost": {
            "estimator": CatBoostClassifier(
                random_seed=RANDOM_STATE,
                verbose=False,
            ),
            "params": {
                "iterations": [100, 200],
                "depth": [4, 6],
                "learning_rate": [0.05, 0.1],
            },
            "n_jobs": -1,
        },
    }

    torch_models = {
        "MLP": {
            "estimator": TorchClassifier(
                model_type="mlp",
                n_features=n_features,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "hidden_dims": [(64, 32), (128, 64)],
                "dropout": [0.2, 0.4],
            },
            "n_jobs": 1,
        },
        "CNN": {
            "estimator": TorchClassifier(
                model_type="cnn",
                n_features=n_features,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "n_channels": [16, 32],
                "kernel_size": [3, 5],
                "dropout": [0.2, 0.4],
            },
            "n_jobs": 1,
        },
        "RNN": {
            "estimator": TorchClassifier(
                model_type="rnn",
                n_features=n_features,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "hidden_size": [32, 64],
                "num_layers": [1, 2],
                "rnn_type": ["lstm", "gru"],
            },
            "n_jobs": 1,
        },
        "Transformer": {
            "estimator": TorchClassifier(
                model_type="transformer",
                n_features=n_features,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "d_model": [32, 64],
                "trans_layers": [2, 3],
                "dropout": [0.1, 0.2],
            },
            "n_jobs": 1,
        },
    }

    results = {}

    best_cv_auc = -1.0
    best_model_obj = None
    best_model_name = None
    best_model_type = "sklearn"
    best_threshold = 0.5
    best_params_used = None

    all_configs = list(sklearn_models.items()) + list(torch_models.items())

    for name, config in all_configs:
        is_torch = name in torch_models
        model_type = "pytorch" if is_torch else "sklearn"

        print(f"\n>>> 训练 {name}  (GridSearchCV 5-fold CV) ...")
        start = time.time()

        try:
            gs = GridSearchCV(
                config["estimator"],
                config["params"],
                cv=cv,
                scoring="roc_auc",
                n_jobs=config["n_jobs"],
                verbose=0,
            )
            gs.fit(X_train_scaled, y_train)

            best_model = gs.best_estimator_
            cv_auc = gs.best_score_

            print(f"    最佳参数: {gs.best_params_}")
            print(f"    最佳 CV AUC: {cv_auc:.4f}")

            if is_torch:
                train_scores, test_scores, opt_th, _ = evaluate_model(
                    best_model.model_,
                    X_train_scaled,
                    y_train,
                    X_test_scaled,
                    y_test,
                    model_type="pytorch",
                )
            else:
                train_scores, test_scores, opt_th, _ = evaluate_model(
                    best_model,
                    X_train_scaled,
                    y_train,
                    X_test_scaled,
                    y_test,
                )

            print(f"    训练集 {format_metrics(train_scores)}")
            print(f"    测试集 {format_metrics(test_scores)}  (阈值={opt_th:.4f})")

            results[name] = test_scores

            if cv_auc > best_cv_auc:
                best_cv_auc = cv_auc
                best_model_obj = best_model
                best_model_name = name
                best_model_type = model_type
                best_threshold = opt_th
                best_params_used = gs.best_params_

        except Exception as e:
            print(f"    {name} 训练失败: {e}")

        print(f"    耗时: {time.time() - start:.1f}秒")

    # ---------- 控制台打印对比 ----------
    print("\n" + "=" * 78)
    print(f"{len(results)} 种模型性能对比 (测试集, 5 项指标)")
    print("=" * 78)

    header = (
        f"{'Model':<20}{'AUC':>10}{'Accuracy':>12}{'F1':>10}"
        f"{'Precision':>12}{'Recall':>10}"
    )
    print(header)
    print("-" * 78)

    for name, scores in sorted(
        results.items(), key=lambda x: x[1]["AUC"], reverse=True
    ):
        print(
            f"{name:<20}{scores['AUC']:>10.4f}{scores['Accuracy']:>12.4f}"
            f"{scores['F1']:>10.4f}{scores['Precision']:>12.4f}"
            f"{scores['Recall']:>10.4f}"
        )

    print("=" * 78)

    if best_model_obj is None:
        print("没有成功训练的模型，程序结束。")
        return

    print(
        f"\n最佳模型（按 CV AUC 选择）: {best_model_name}, "
        f"CV AUC = {best_cv_auc:.4f}, 阈值 = {best_threshold:.4f}"
    )

    # ---------- 全样本预测概率 ----------
    if best_model_type == "pytorch":
        device = next(best_model_obj.model_.parameters()).device
        best_model_obj.model_.eval()

        with torch.no_grad():
            total_proba = (
                best_model_obj.model_(
                    torch.tensor(X_total_scaled, dtype=torch.float32).to(device)
                )
                .cpu()
                .numpy()
                .flatten()
            )
        total_proba = np.clip(total_proba, 0, 1)

    else:
        total_proba = np.clip(
            best_model_obj.predict_proba(X_total_scaled)[:, 1], 0, 1
        )

    # ============================================================
    # 表 1：五张指标对比
    # ============================================================
    perf_df = pd.DataFrame(results).T.reset_index().rename(
        columns={"index": "Model"}
    )
    perf_df = (
        perf_df[["Model"] + METRIC_KEYS]
        .sort_values("AUC", ascending=False)
        .reset_index(drop=True)
    )

    for col in METRIC_KEYS:
        perf_df[col] = perf_df[col].round(4)

    perf_csv = os.path.join(save_dir, "1_五张指标对比.csv")
    perf_df.to_csv(perf_csv, index=False, encoding="utf-8-sig")
    print(f"\n[输出表 1] 五张指标对比: {perf_csv}")

    # ============================================================
    # 表 2：最优模型全样本易发性预测值
    # ============================================================
    best_result_df = first_three_cols.copy()
    best_result_df["易发性概率"] = np.round(total_proba, 4)

    best_csv = os.path.join(save_dir, "2_最优模型结果.csv")
    best_result_df.to_csv(best_csv, index=False, encoding="utf-8-sig")

    print(f"[输出表 2] 海底滑坡易发性评价预测值结果: {best_csv}")
    print(
        f"          (最优模型: {best_model_name}, 阈值: {best_threshold:.4f}, "
        f"概率范围: [{total_proba.min():.4f}, {total_proba.max():.4f}])"
    )

    # ============================================================
    # 表 3：SHAP 特征重要性
    # ============================================================
    shap_model = (
        best_model_obj.model_ if best_model_type == "pytorch" else best_model_obj
    )

    shap_df = shap_feature_importance(
        best_model=shap_model,
        best_model_name=best_model_name,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_test=y_test,
        feature_names=kept_features,
        model_type=best_model_type,
        max_background=100,
        max_explain=200,
    )

    if shap_df is None:
        shap_df = pd.DataFrame(columns=["Rank", "Feature", "Mean_Abs_SHAP"])
        print("    SHAP 计算失败或为空，已输出空表 3。")

    shap_csv = os.path.join(save_dir, "3_SHAP特征重要性排序.csv")
    shap_df.to_csv(shap_csv, index=False, encoding="utf-8-sig")
    print(f"[输出表 3] SHAP 特征重要性排序: {shap_csv}")

    print("\n" + "=" * 78)
    print("处理完成！仅输出 3 个 CSV 文件：")
    print("  1_五张指标对比.csv")
    print("  2_最优模型结果.csv")
    print("  3_SHAP特征重要性排序.csv")
    print(f"输出目录: {save_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main(
        input_csv=r"C:\Users\admin\Desktop\floating-wind-SLS\Submarine landslide susceptibility\输入\样本点.csv",
        output_dir=r"C:\Users\admin\Desktop\floating-wind-SLS\Submarine landslide susceptibility\输出",
        encoding="gbk",
    )