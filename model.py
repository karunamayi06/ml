import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)

st.title("Model Performance Checker with Visualizations")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])


def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset", df.head())

    target_col = st.selectbox("Select Target Column", df.columns)
    feature_cols = st.multiselect(
        "Select Feature Columns", [c for c in df.columns if c != target_col]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Ridge Classifier": RidgeClassifier(),
        "SGD Classifier": SGDClassifier(),
        "Support Vector Classifier (SVC)": SVC(probability=True),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Bagging Classifier": BaggingClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis(),
    }
    selected_models = st.multiselect(
        "Select Models to Evaluate", list(models.keys()), default=list(models.keys())
    )

    if target_col and feature_cols and selected_models and st.button("Run Evaluation"):
        # Drop rows with missing target
        data = df[feature_cols + [target_col]].dropna(subset=[target_col])
        X = data[feature_cols]

        # Encode target (works for strings and numbers)
        le = LabelEncoder()
        y = le.fit_transform(data[target_col])

        if len(le.classes_) > 50:
            st.warning(
                f"Target has {len(le.classes_)} unique values — this looks like "
                "regression, not classification. Pick a different target."
            )
            st.stop()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:  # classes too small to stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        rows, cms = [], {}
        progress = st.progress(0.0)

        for i, name in enumerate(selected_models):
            pipe = Pipeline([
                ("prep", build_preprocessor(X_train)),
                ("model", models[name]),
            ])
            try:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
            except Exception as e:
                st.warning(f"Skipped {name}: {e}")
                progress.progress((i + 1) / len(selected_models))
                continue

            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            })
            cms[name] = confusion_matrix(y_test, y_pred, labels=range(len(le.classes_)))
            progress.progress((i + 1) / len(selected_models))

        st.session_state["metrics_df"] = pd.DataFrame(rows)
        st.session_state["cms"] = cms
        st.session_state["classes"] = le.classes_

    # ---- Display results (persist across reruns) ----
    if "metrics_df" in st.session_state and not st.session_state["metrics_df"].empty:
        metrics_df = st.session_state["metrics_df"]

        st.subheader("Performance Metrics")
        st.dataframe(metrics_df.sort_values("F1-Score", ascending=False))

        st.subheader("Model Performance Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics_df.set_index("Model")[["Accuracy", "F1-Score"]].plot(
            kind="bar", ax=ax, color=["skyblue", "orange"]
        )
        plt.title("Accuracy & F1-Score Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Confusion Matrix")
        cm_model = st.selectbox("Choose a model", list(st.session_state["cms"].keys()))
        classes = st.session_state["classes"]
        fig, ax = plt.subplots()
        sns.heatmap(
            st.session_state["cms"][cm_model], annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {cm_model}")
        st.pyplot(fig)
