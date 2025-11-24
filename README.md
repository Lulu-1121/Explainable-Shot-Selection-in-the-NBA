# Explainable Shot-Selection in the NBA

**Linking player tracking data with decision-making transparency**

---

## 🚀 Overview

This project builds an **explainable machine learning framework** that evaluates **whether a shot attempt in the NBA is optimal**—not just whether it goes in.
It combines **play-by-play data**, **contextual features**, and **XAI techniques (SHAP + counterfactual “what-if” analysis)** to deliver **real-time decision support for coaches and analysts**.

> Unlike traditional black-box models that focus only on prediction, this project emphasizes **transparency, interpretability, and tactical insight** — enabling users to understand *why* a specific shot was taken and how different contextual factors may have changed the outcome.

---

## 📸 Demo (Web App Screenshot)

<img width="2507" height="1311" alt="image" src="https://github.com/user-attachments/assets/8b1568f8-e7ff-4bc1-b1b3-6b7f3c39a1d9" />

---

## 🧠 Key Questions

| Question                                      | Answered By                      |
| --------------------------------------------- | -------------------------------- |
| Was this shot a high-value attempt?           | LightGBM classification          |
| Why was this decision made?                   | SHAP feature attributions        |
| Would changing conditions alter the decision? | Counterfactual “what-if” sliders |
| Which features impact shot selection most?    | Feature importance heatmaps      |

---

## 🔍 Motivation

* Every shot has **opportunity cost**: possession, fatigue, lineup choices
* Current analytics often **predict make/miss** without explaining **decision quality**
* Coaches and analysts need **transparent, trustable models** to improve strategy
* This project bridges **XAI and sports analytics**, supporting real-time tactical evaluation

---

## 🧰 Data & Methods

| Component          | Description                                                         |
| ------------------ | ------------------------------------------------------------------- |
| **Dataset**        | NBA SQLite database (Kaggle: 65k+ games, 4.8M+ rows)                |
| **Features**       | defender proximity, score margin, period, home/away flag, shot type |
| **Model**          | LightGBM (binary classification)                                    |
| **Explainability** | SHAP values + counterfactual generation                             |
| **UI**             | Streamlit interactive dashboard                                     |

---

## 🧪 Example Use Case

### 🧾 Original Context

> Given current game context, was the shot a high-value attempt?
> The model predicts the probability and shows **which features drove the decision**.

### 🔁 What-if Scenario

> What if defender was farther away?
> What if score margin was different?
> What if this was a three-point attempt?

The app recomputes shot value and **updates SHAP explanations instantly**.

---

## ⚙️ Tech Stack

```
Python 3.10  
Streamlit  
LightGBM  
Scikit-learn  
Pandas / NumPy  
SHAP  
Plotly  
SQLite  
```

---

## 📂 Project Structure

```
├── src/
│   ├── extract_shots.py         # Load & preprocess SQLite data
│   ├── train_model.py           # Train LightGBM model
│   ├── app_streamlit.py        # Webapp (XAI + UI)
│
├── data/
│   ├── processed/               # Final parquet data for app
│
├── models/
│   ├── lgbm_shot_model.pkl      # Trained model
│
├── requirements.txt
├── runtime.txt                  # Python version for Streamlit Cloud
└── README.md
```

---

## 🖥️ Deployment

The app is fully deployed with **Streamlit Cloud**, enabling one-click access without running any local code.

> 🔗 **Live App**: [https://explainable-shot-selection-in-the-nba.streamlit.app](https://explainable-shot-selection-in-the-nba.streamlit.app)

---

## 🌟 Key Innovations

✔️ Moves beyond “make/miss” prediction — **evaluates decision quality**
✔️ Integrates **XAI (SHAP) + counterfactual reasoning** in sports analytics
✔️ Enables **interactive tactical review** for coaching staff
✔️ Uses **real NBA tracking data** for realistic modeling
✔️ Ready to extend for **player-specific models** or **team scouting reports**

---

## 📌 Future Extensions

* Player-level shot tendency modeling
* Lineup recommendation engine
* Clutch-time optimization analysis
* Real-time API for live games
* Comparison of human vs model decisions

---

## 👤 Author

**Hongyi Duan**
MIDS Graduate Student – Duke University
Focus: Responsible AI & Sports Analytics

---

Let me know when you want:

* Professional research poster version (for final pitch)
* Polished video script (≤5 min, required by rubric)
* PDF report / Expo presentation template
