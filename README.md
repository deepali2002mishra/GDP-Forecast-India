#India GDP Forecasting Project (2025–2030)

This project predicts India’s **national and sectoral GDP** using a hybrid modeling approach combining **SARIMAX** and **XGBoost**, enriched with realistic economic scenario simulations. The goal is to assist in data-driven investment planning and economic policymaking through explainable machine learning and statistical forecasting.

---

## 📁 Project Structure

GDP/ ├── Dashboard.py # Streamlit dashboard (main UI) ├── scripts/ # Preprocessing, modeling, and forecasting logic ├── models/ # Trained SARIMAX and XGBoost models ├── results/ # Forecast outputs, plots, and reports ├── data/ # Raw and processed datasets ├── requirements.txt # Dependencies └── README.md # You are here!

## 🌟 Key Features

- ✅ Forecast India's **National GDP Growth** from 2025 to 2030
- ✅ Sectoral Forecasting: **Agriculture** 🌾 and **IT Sector** 💻
- ✅ Simulate Economic Scenarios: **Baseline**, **Reform Acceleration**, and **External Crisis**
- ✅ **Hybrid Modeling** using SARIMAX (for trends) + XGBoost (for residual learning)
- ✅ **SHAP Explainability** for indicator impact analysis
- ✅ **Exportable Reports**: State-wise investment rationale and sector forecasts
- ✅ **Interactive Streamlit Dashboard** for easy navigation and presentation

---

## 🚀 How to Run Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/deepali2002mishra/GDP-Forecast-India.git
   cd GDP-Forecast-India

Install dependencies
pip install -r requirements.txt

Launch the dashboard
streamlit run Dashboard.py

📊 Outputs
📈 National GDP forecasts (2025–2030)

🌦 Scenario-wise projections: Reform, Crisis, Baseline
🌾 State-wise agriculture growth and investment suggestions
💻 IT sector GDP forecasts
💡 SHAP plots showing key economic indicator contributions

📌 Technologies Used
Python 3.11
Streamlit
SARIMAX (Statsmodels)
XGBoost
SHAP
Pandas, NumPy, Matplotlib, Scikit-learn

🤝 Team Members
