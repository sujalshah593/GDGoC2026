# ⚖️ FairSight — AI Bias Detection & Mitigation Dashboard

> Detect, visualize, and mitigate algorithmic bias in your datasets — powered by **AIF360**, **Scikit-learn**, and **Gemini AI**.

---

## 📌 Overview

FairSight is an interactive Streamlit dashboard that helps data scientists, ML engineers, and researchers audit their datasets for fairness issues. It measures **Disparate Impact** across protected groups (e.g. gender, race, age), applies **Reweighing** mitigation, and generates an AI-powered explanation of the bias using Google's Gemini models.

---

## 🗂 Project Structure

```
├── app.py           # Main Streamlit application (UI + analysis pipeline)
├── bias_utils.py    # Core fairness utilities (preprocessing, AIF360 wrappers)
└── README.md
```

### `app.py`
The entry point for the Streamlit app. Responsibilities include:
- File upload and parsing (CSV / JSON)
- Dataset configuration (target column, protected attribute)
- Validation and auto binary conversion of columns
- Orchestrating the full AIF360 fairness pipeline
- Rendering all UI: metrics, charts, status banners, AI explanation
- Calling Gemini API for natural-language bias explanation

### `bias_utils.py`
A utility module that wraps AIF360 and pandas logic. Exposes these functions:

| Function | Description |
|---|---|
| `preprocess_data(df, target_col, protected_col)` | Cleans data, encodes target & protected columns, one-hot encodes remaining features |
| `prepare_dataset(df, label_col, protected_col)` | Wraps a DataFrame into an AIF360 `BinaryLabelDataset` |
| `measure_bias(dataset, protected_col, priv, unpriv)` | Computes **Disparate Impact** using `BinaryLabelDatasetMetric` |
| `mitigate_bias(dataset, protected_col, priv, unpriv)` | Applies **Reweighing** algorithm via AIF360 |
| `group_outcome_rates(dataset, protected_col, use_weights)` | Returns per-group positive outcome rates (weighted or unweighted) |
| `map_group_value(col, val)` | Maps binary encoded values back to human-readable labels (Male/Female, Majority/Others, etc.) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
<<<<<<< HEAD
git clone https://github.com/your-username/fairsight.git
cd fairsight
=======
git clone https://github.com/Denish2410/GDGOC
cd GDGOC
cd GDGoC_2026
>>>>>>> 85eba766afec004158e3b66480ec2941b9a6fb15
```

### 2. Install dependencies

```bash
pip install streamlit pandas matplotlib scikit-learn aif360 google-genai
```

> **Note:** AIF360 may require additional system dependencies. Refer to the [AIF360 installation guide](https://github.com/Trusted-AI/AIF360#installation) if you run into issues.

### 3. Set up Gemini API

FairSight uses Google Gemini for AI-generated explanations. Set your API key:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create a `.env` file and load it using `python-dotenv`.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🧭 How to Use

**Step 1 — Upload Dataset**
Upload a `.csv` or `.json` file. Common test datasets include Adult Income, COMPAS Recidivism, and German Credit.

**Step 2 — Configure Analysis**
- Select the **Target Column** (the outcome to predict, e.g. `income`, `loan_approved`). Must be binary or convertible to binary.
- Select the **Protected Attribute** (e.g. `gender`, `race`, `age`) — the column to check for bias.
- Choose which group is **Privileged** (e.g. Male, Majority).

**Step 3 — Run Analysis**
Click **⚡ Run Bias Analysis**. The dashboard will:
1. Preprocess and encode the data
2. Train a Logistic Regression model on the original data
3. Measure **Disparate Impact (DI)** before mitigation
4. Apply **Reweighing** to adjust instance weights
5. Retrain the model on reweighted data
6. Measure DI after mitigation
7. Display group outcome rates, gap analysis, and an AI explanation

---

## 📊 Key Concepts

### Disparate Impact (DI)
The primary fairness metric used. It is the ratio of positive outcome rates between the unprivileged and privileged groups:

```
DI = P(outcome=1 | unprivileged) / P(outcome=1 | privileged)
```

| DI Value | Interpretation |
|---|---|
| 0.8 – 1.25 | ✅ Fair (acceptable range) |
| 0.6 – 0.8 | ⚠️ Moderate bias |
| < 0.6 or > 1.25 | ❌ High bias |

A DI of **1.0** means both groups have equal positive outcome rates — perfect fairness.

### Reweighing (Mitigation)
A pre-processing technique from AIF360 that assigns different weights to training instances to compensate for bias in the data. It does not alter the data itself — only influences how the model learns from it.

### Outcome Gap
The absolute difference in positive outcome rates between the privileged and unprivileged groups after mitigation. A gap below **0.1** is considered acceptable.

---

## 🤖 AI Explanation

After analysis, FairSight calls the **Google Gemini API** to generate a plain-language explanation covering:
1. What the detected bias means
2. Why it might have occurred in the data
3. Whether the mitigation strategy helped
4. Recommended actions to take

Model selection is automatic:
- **`gemini-2.5-pro`** — used when outcome gap > 0.2 (high bias, deeper explanation needed)
- **`gemini-2.5-flash`** — used for moderate or low bias cases

---

## ⚠️ Limitations

- Only supports **binary classification** tasks. Multi-class targets are auto-converted via median split, which may lose meaning.
- Only supports **binary protected attributes**. Multi-valued attributes (e.g. race with many categories) are collapsed to majority vs. others.
- Mitigation uses only **Reweighing**. Other strategies (e.g. adversarial debiasing, post-processing) are not yet implemented.
- Bias metrics reflect patterns in the **uploaded dataset** — they do not guarantee fairness of a deployed model on unseen data.
- The AI explanation is generated by an external LLM and should be reviewed critically.

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI framework |
| [AIF360](https://github.com/Trusted-AI/AIF360) | Fairness metrics and mitigation algorithms |
| [Scikit-learn](https://scikit-learn.org) | Logistic Regression classifier |
| [Pandas](https://pandas.pydata.org) | Data manipulation |
| [Matplotlib](https://matplotlib.org) | Charts and visualizations |
| [Google Gemini](https://ai.google.dev) | AI-generated bias explanations |

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute it with attribution.

---

## 🙋 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
