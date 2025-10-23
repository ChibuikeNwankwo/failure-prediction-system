"""
Streamlit app for Predictive Maintenance — Two-Stage Model
Run: streamlit run app_streamlit.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
from functions import clean_col_names
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

st.set_page_config(page_title="Equipment Failure Prediction App", layout="wide")

# ============== LOAD MODELS & ENCODERS ==============
stage1 = joblib.load('models/voting_stage1_model.pkl')
stage2 = joblib.load('models/voting_stage2_model.pkl')
thresholds = joblib.load('models/voting_stage1_thresholds.pkl')

type_encoder = joblib.load("models/type_encoder.pkl")      
failure_encoder = joblib.load("models/failure_type_encoder.pkl")

# ============== DEFINE NUMERIC RANGES ==============
feature_ranges = {
    "Air temperature [K]": (293, 310),
    "Process temperature [K]": (300, 320),
    "Rotational speed [rpm]": (1000, 3000),
    "Torque [Nm]": (1, 100),
    "Tool wear [min]": (0, 270)
}

# ============== PAGE CONFIG ==============
st.set_page_config(page_title="Predictive Maintenance System", layout="wide")
st.title("🏭 Predictive Maintenance Intelligence Dashboard")

# ============== SIDEBAR NAVIGATION ==============
view = st.sidebar.radio("Navigation", ["📊 Analytics Dashboard", "⚙️ Live Prediction"])

# ====================================================================
# 📊 ANALYTICS DASHBOARD (EDA + MODEL PERFORMANCE)
# ====================================================================
if view == "📊 Analytics Dashboard":
    st.header("📈 Data & Model Insights")
    tab1, tab2 = st.tabs(["Exploratory Data Analysis", "Model Performance"])

    # ========== TAB 1: EDA ==========
    with tab1:
        st.subheader("Feature Distributions (Training Data)")
        df_train = pd.read_csv("data/eval_data.csv", index_col = 0)

        # ----------------------------------
        # 🔸 Toggle for Plot Mode
        # ----------------------------------
        view_mode = st.radio("Select plot type:", ["Static (Seaborn/Matplotlib)", "Interactive (Plotly)"])

        # ----------------------------------
        # 🔹 NUMERIC FEATURE DISTRIBUTIONS
        # ----------------------------------
        st.subheader("📈 Numeric Feature Distributions")

        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['UDI', 'Target']]

        if view_mode == "Static (Seaborn/Matplotlib)":
            selected_num = st.selectbox("Select numeric feature:", numeric_cols, key="num_static")
            fig, ax = plt.subplots()
            sns.histplot(df_train[selected_num], kde=True, ax=ax, color='steelblue')
            ax.set_title(f"Distribution of {selected_num}", fontsize=14, fontweight="bold")
            st.pyplot(fig)
        else:
            selected_num = st.selectbox("Select numeric feature:", numeric_cols, key="num_interactive")
            fig = px.histogram(df_train, x=selected_num, nbins=30, title=f"Distribution of {selected_num}",
                               marginal='box', color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------
        # 🔹 CATEGORICAL FEATURE DISTRIBUTIONS
        # ----------------------------------
        st.subheader("🧮 Categorical Feature Distributions")

        cat_options = ["Target", "Type", "Failure Type (only failures)", "Failure Type per Type"]
        cat_selected = st.selectbox("Select categorical plot:", cat_options)

        if view_mode == "Static (Seaborn/Matplotlib)":
            fig, ax = plt.subplots()

            if cat_selected == "Target":
                sns.countplot(x='Target', data=df_train, ax=ax, palette='pastel')
                ax.set_title("Count of Target Classes")
            elif cat_selected == "Type":
                sns.countplot(x='Type', data=df_train, ax=ax, palette='muted')
                ax.set_title("Count of Machine Types")
            elif cat_selected == "Failure Type (only failures)":
                sns.countplot(x='Failure Type', data=df_train[df_train['Target'] == 1], ax=ax, palette='Set2')
                ax.set_title("Failure Type Distribution (Only Failures)")
            elif cat_selected == "Failure Type per Type":
                sns.countplot(x='Type', hue='Failure Type', data=df_train[df_train['Target'] == 1], ax=ax, palette='Set1')
                ax.set_title("Failure Type per Machine Type")

            plt.xticks(rotation=45)
            st.pyplot(fig)

        else:
            if cat_selected == "Target":
                fig = px.histogram(df_train, x='Target', color='Target', title="Count of Target Classes",
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
            elif cat_selected == "Type":
                fig = px.histogram(df_train, x='Type', color='Type', title="Count of Machine Types",
                                   color_discrete_sequence=px.colors.qualitative.Set2)
            elif cat_selected == "Failure Type (only failures)":
                failures_only = df_train[df_train['Target'] == 1]
                fig = px.histogram(failures_only, x='Failure Type', color='Failure Type',
                                   title="Failure Type Distribution (Only Failures)",
                                   color_discrete_sequence=px.colors.qualitative.Bold)
            elif cat_selected == "Failure Type per Type":
                failures_only = df_train[df_train['Target'] == 1]
                fig = px.histogram(failures_only, x='Type', color='Failure Type', barmode='group',
                                   title="Failure Type per Machine Type",
                                   color_discrete_sequence=px.colors.qualitative.Safe)

            fig.update_layout(bargap=0.2, xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------
        # 🔹 SCATTERPLOTS (ONLY FAILURES)
        # ----------------------------------
        st.subheader("🎯 Scatterplots of Numeric Features by Failure Type (Only Failures)")

        failures_only = df_train[df_train["Target"] == 1].copy()

        x_feat = st.selectbox("Select X-axis feature:", numeric_cols, key="scatter_x")
        y_feat = st.selectbox("Select Y-axis feature:", numeric_cols, key="scatter_y")

        if view_mode == "Static (Seaborn/Matplotlib)":
            fig, ax = plt.subplots()
            sns.scatterplot(data=failures_only, x=x_feat, y=y_feat, hue="Failure Type", ax=ax, s=50)
            ax.set_title(f"{x_feat} vs {y_feat} by Failure Type (Only Failures)")
            st.pyplot(fig)
        else:
            fig = px.scatter(failures_only, x=x_feat, y=y_feat, color='Failure Type', symbol='Failure Type',
                             title=f"{x_feat} vs {y_feat} by Failure Type (Only Failures)",
                             hover_data=['Type'] if 'Type' in failures_only.columns else None,
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='black')))
            fig.update_layout(legend=dict(title="Failure Type", orientation="h", y=-0.2, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------
        # 🔹 BOX PLOTS BY TARGET
        # ----------------------------------
        st.subheader("📦 Boxplots of Numeric Features by Target")

        selected_box = st.selectbox("Select numeric feature for boxplot:", numeric_cols, key="box_feature")

        if view_mode == "Static (Seaborn/Matplotlib)":
            fig, ax = plt.subplots()
            sns.boxplot(x='Target', y=selected_box, data=df_train, palette='pastel', ax=ax)
            sns.stripplot(x='Target', y=selected_box, data=df_train, color='black', alpha=0.4, ax=ax)
            ax.set_title(f"{selected_box} Distribution by Target")
            st.pyplot(fig)
        else:
            fig = px.box(df_train, x='Target', y=selected_box, color='Target', points='all',
                         title=f"{selected_box} Distribution by Target (Failure vs No Failure)",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(xaxis_title="Target (0 = No Failure, 1 = Failure)", yaxis_title=selected_box)
            st.plotly_chart(fig, use_container_width=True)


        
    # ========== TAB 2: MODEL PERFORMANCE ==========
    with tab2:
        df = clean_col_names(df_train)

        expected_order = [
            "Air_temperature_K","Process_temperature_K",
            "Rotational_speed_rpm","Torque_Nm","Tool_wear_min",
            "Temp_diff","Torque_per_rpm","Wear_per_rpm","Type"
        ]
        df = df[expected_order + ["Target","Failure_Type"]]  # keep targets for metrics

        # --- Stage 1 predictions ---
        X1 = df[expected_order]
        if 'Type' in X1.columns:
            X1['Type'] = type_encoder.transform(X1['Type'])
        y1_true = df["Target"]
        y1_pred = stage1.predict(X1)
        y1_proba = stage1.predict_proba(X1)[:,1]

        # Metrics for Stage 1
        acc1 = accuracy_score(y1_true, y1_pred)
        prec1 = precision_score(y1_true, y1_pred)
        rec1 = recall_score(y1_true, y1_pred)
        f11 = f1_score(y1_true, y1_pred)
        roc_auc1 = roc_auc_score(y1_true, y1_proba)
        cm1 = confusion_matrix(y1_true, y1_pred)

        # --- Stage 2 predictions (only where failure predicted) ---
        idx_fail_pred = (y1_pred == 1)&(y1_true == 1)
        X2 = X1[idx_fail_pred]
        y2_true = df.loc[idx_fail_pred, "Failure_Type"]
        y2_true_enc = failure_encoder.transform(y2_true)

        y2_pred_enc = stage2.predict(X2)
        y2_pred = failure_encoder.inverse_transform(y2_pred_enc)

        # Metrics for Stage 2
        report2 = classification_report(y2_true_enc, y2_pred_enc, target_names=failure_encoder.classes_, output_dict=True)
        cm2 = confusion_matrix(y2_true_enc, y2_pred_enc)

        # --- Display Metrics ---
        st.markdown("### Stage 1: Failure Detection")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc1:.3f}")
        c2.metric("Precision", f"{prec1:.3f}")
        c3.metric("Recall", f"{rec1:.3f}")
        c4.metric("F1-Score", f"{f11:.3f}")
        c5.metric("ROC-AUC", f"{roc_auc1:.3f}")

        fig, ax = plt.subplots()
        sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Stage 1 Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Stage 2: Failure Cause Classification")
        st.dataframe(pd.DataFrame(report2).T)

        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=ax)
        ax.set_title("Stage 2 Confusion Matrix")
        ax.set_xlabel("Predicted Cause"); ax.set_ylabel("Actual Cause")
        st.pyplot(fig)
# ====================================================================
# ⚙️ LIVE PREDICTION DASHBOARD
# ====================================================================
elif view == "⚙️ Live Prediction":
    st.header("⚙️ Real-time Failure Prediction")

    # --- Input Section ---
    st.subheader("Input Parameters")

    # Type dropdown (categorical)
    type_options = list(type_encoder.classes_)
    Type = st.selectbox("Type", options=type_options)

    # Numeric Inputs
    input_data = {}
    for feature, (fmin, fmax) in feature_ranges.items():
        value = st.number_input(
            feature,
            min_value=float(fmin),
            max_value=float(fmax),
            value=float((fmin + fmax) / 2),
            step=0.1
        )
        input_data[feature] = value

    # Encode 'Type' only for model input
    input_data_encoded = input_data.copy()
    input_data_encoded["Type"] = type_encoder.transform([Type])[0]

    df_in = pd.DataFrame([input_data_encoded])

    if st.button("Predict"):
        # --- Feature Engineering ---
        df_in['Temp_diff'] = df_in['Process temperature [K]'] - df_in['Air temperature [K]']
        df_in['Torque_per_rpm'] = df_in['Torque [Nm]'] / (df_in['Rotational speed [rpm]'] + 1e-6)
        df_in['Wear_per_rpm'] = df_in['Tool wear [min]'] / (df_in['Rotational speed [rpm]'] + 1e-6)
        df_in = clean_col_names(df_in)

        expected_order = [
            'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm',
            'Torque_Nm', 'Tool_wear_min', 'Temp_diff', 'Torque_per_rpm', 'Wear_per_rpm', 'Type'
        ]
        df_in = df_in[expected_order]

        # --- Stage 1 Prediction ---
        X1 = df_in
        proba = stage1.predict_proba(X1)[:, 1][0]
        st.metric("Failure Probability", f"{proba:.3f}")

        if proba >= 0.51:
            st.error("Prediction: FAILURE ⚠️")

            # Stage 2 Prediction
            df_in2 = df_in.copy()
            ft = stage2.predict(df_in2)[0]
            ft = failure_encoder.inverse_transform([ft])[0]
            st.success(f"Predicted Failure Cause: {ft}")

        else:
            st.success("Prediction: NO FAILURE ✅")
