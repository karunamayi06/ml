import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Streamlit App
st.title("Model Performance Checker with Visualizations")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset", df.head())

    # Feature and Target selection
    target_col = st.selectbox("Select Target Column", df.columns)
    feature_cols = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target_col])

    if target_col and feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models to Evaluate
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
        selected_models = st.multiselect("Select Models to Evaluate", models.keys(), default=list(models.keys()))

        results = {}
        metrics_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])

        for name in selected_models:
            model = models[name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            new_row = {
                "Model": name,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)


            # Confusion Matrix Visualization
            if st.checkbox(f"Show Confusion Matrix for {name}"):
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=df[target_col].unique(), yticklabels=df[target_col].unique())
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title(f"Confusion Matrix - {name}")
                st.pyplot(fig)

        # Display Metrics
        st.write("Performance Metrics:")
        st.dataframe(metrics_df)

        # Bar Chart for Comparison
        if not metrics_df.empty:
            st.subheader("Model Performance Comparison")
            fig, ax = plt.subplots(figsize=(8, 4))
            metrics_df.set_index("Model")[["Accuracy", "F1-Score"]].plot(kind="bar", ax=ax, color=["skyblue", "orange"])
            plt.title("Accuracy & F1-Score Comparison")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)
