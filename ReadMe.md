# ⚙️ 2-Stage Failure Prediction & Root Cause Analysis System

## 🚀 Overview

This project demonstrates a **real-world, production-style machine learning system** that detects failures and predicts their root causes in two stages.  
It showcases the ability to combine **predictive modeling, business reasoning, and analytics visualization** into one cohesive solution.

Built with **modular design and scalability in mind**, the project mimics how predictive maintenance or quality-control systems work in manufacturing, logistics, and industrial IoT. The design can be adapted to **other domains** like **finance (fraud → fraud type)** or **healthcare (disease → subtype)**.

> **Goal:** Accurately identify which units are likely to fail — and explain *why* — enabling data-driven preventive actions.

---

## 🧩 Project Architecture

### 🥇 Stage 1 — Failure Detection
A **binary classification model** predicts whether a product, process, or component will fail.  
This stage is optimized for high recall to ensure potential failures are rarely missed.

### 🥈 Stage 2 — Root Cause Prediction  
A **multiclass model** predicts the *cause* of each correctly identified failure (e.g., electrical, mechanical, calibration).  
This helps teams understand underlying failure patterns and target maintenance efforts efficiently.

Both stages are linked seamlessly:
- Stage 1 filters high-risk cases  
- Stage 2 diagnoses the specific cause for those cases  


---

## 🎯 Key Objectives
✅ Detect failures early with high precision  
✅ Predict failure causes for actionable insights  
✅ Provide an analytics dashboard for visual inspection of model performance  
✅ Enable scalability to 100k+ records without retraining from scratch  

---

## 🧠 How It Works

| Step | Process | Description |
|------|----------|-------------|
| **1** | Data Ingestion | Dataset is loaded, cleaned, and split into train/test sets |
| **2** | Stage 1 Modeling | Trains a binary classifier for failure detection |
| **3** | Stage 1 Evaluation | Computes accuracy, precision, recall, F1, and confusion matrix |
| **4** | Stage 2 Dataset Creation | Filters correctly predicted failures for root cause modeling |
| **5** | Stage 2 Modeling | Trains a cause-classification model |
| **6** | Stage 2 Evaluation | Calculates per-cause metrics and macro/micro scores |
| **7** | Visualization | Results are displayed in a dynamic analytics dashboard |

---

## 💼 Business Impact

| Problem | Traditional Approach | This Solution |
|----------|----------------------|----------------|
| Late failure detection | Manual inspection & lagging indicators | Predicts failures *before* they happen |
| Unknown root causes | Reactive troubleshooting | Automated cause prediction per failure |
| Limited visibility | Disconnected reports | Unified dashboard for data & insights |
| Hard to scale | Ad-hoc scripts | Modular 2-stage pipeline, ready for deployment |

This system directly supports **operational efficiency**, **cost reduction**, and **risk mitigation** — making it applicable to:
- Manufacturing & production lines  
- Predictive maintenance  
- Quality control analytics  
- Process optimization in industrial settings  

---

## 📊 Results & Insights

| Metric | Stage 1 (Failure Detection) | Stage 2 (Cause Classification) |
|--------|-----------------------------|-------------------------------|
| Accuracy | 0.998 | 0.981 |
| Precision | 0.973 | 0.977 |
| Recall | 0.970 | 0.975 |
| F1-Score | 0.971 | 0.977 |

*(Results are dynamically computed when the model is tested — no hardcoded values.)*

Additional visuals in the dashboard include:
- Confusion Matrices for both stages  
- Failure Distribution and Class Frequency plots  
- Stage Transition Summary (how many predicted failures proceed to Stage 2)  

---

## 📈 Analytics Dashboard

The project includes a **Streamlit-powered dashboard** that visualizes:
- Model performance metrics
- Feature importances
- Failure vs Non-Failure distribution
- Root cause prediction summary
- Comparison between actual and predicted labels

The dashboard **auto-loads results from model evaluations** and displays them interactively — **no need for users to upload datasets**.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-------------|
| **Programming Language** | Python 3.10+ |
| **Data Science** | pandas, numpy, scikit-learn |
| **Visualization** | matplotlib, seaborn, plotly |
| **Dashboard** | Streamlit |
| **Model Serving** | FastAPI (optional extension) |
| **Deployment Ready** | Docker-compatible structure |

---

## ⚙️ Installation & Usage

```bash
# 1️⃣ Clone the repository
git clone https://github.com/ChibuikeNwankwo/failure-prediction-system.git
cd failure-prediction-system

# 2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate  # (use venv\Scripts\activate on Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Launch the dashboard
streamlit run app.py

# 🔧 Predictive Maintenance Dashboard
```
## 🔍 Example Output

### Stage 1
Predicts whether a record will fail.

### Stage 2
Explains why — identifying the underlying root cause.

---

## 🧮 Evaluation Logic (Automatic Computation)

No values are hardcoded.  
Metrics are computed by testing each stage on the **full dataset**:

- **Stage 1:** Metrics are computed from **all samples**.  
- **Stage 2:** Metrics are computed on the subset of **correctly predicted failures**.  
- The **transition between stages** is automatically logged and displayed in the dashboard.

---

## 🌍 Business Value

This system demonstrates how **multi-stage AI pipelines** can enhance reliability and interpretability:

- 💰 Saves operational cost by detecting failures early  
- ⚙️ Reduces downtime by highlighting actionable root causes  
- 📈 Scales seamlessly with additional data  
- 🏭 Mimics real-world **industrial predictive maintenance** setups  

---

## 🚀 Portfolio-Grade Features

This project showcases:

- 🔍 Advanced **data preprocessing**
- 🧠 Multi-stage **machine learning pipeline**
- 📊 Automated **evaluation pipeline design**
- 🖥️ Interactive **Streamlit analytics dashboard**

---

## 👨‍💻 Author

**Chibuike Nwankwo**  

 – [LinkedIn](https://www.linkedin.com/in/chibuike-nwankwo55 )| [GitHub Portfolio](https://github.com/ChibuikeNwankwo)| [Fiverr](https://www.fiverr.com/krisanalytics)

> “A practical demonstration of real-world ML system design — modular, interpretable, and business-aligned.”
