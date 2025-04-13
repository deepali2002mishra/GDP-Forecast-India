import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- PAGE CONFIG ----
st.set_page_config(page_title="India GDP Forecast Dashboard", layout="wide")
plt.rcParams.update({'font.size': 9})  # Shrink plot font

# ---- HELPERS ----
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df

def get_forecast_column(df):
    forecast_cols = [col for col in df.columns if "forecast" in col.lower()]
    if forecast_cols:
        return forecast_cols[0]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return numeric_cols[-1]

# ---- LOAD CLEANED DATA ----
baseline = clean_columns(pd.read_csv("../../results/national/gdp_forecast_baseline_2025_2026.csv"))
reform = clean_columns(pd.read_csv("../../results/national/gdp_forecast_reform_2027_2030.csv"))
crisis = clean_columns(pd.read_csv("../../results/national/gdp_forecast_crisis_2027_2030.csv"))
mixed = clean_columns(pd.read_csv("../../results/national/gdp_forecast_mixed_2027_2030.csv"))
sarimax_df = pd.read_csv("../../data/processed/sarimax_predictions.csv")
df = pd.read_csv("../../data/processed/processed_data.csv")

# Convert Year columns to integers
for frame in [baseline, reform, crisis, mixed, sarimax_df, df]:
    frame["Year"] = frame["Year"].astype(int)

# Assign scenario labels
baseline['Scenario'] = 'Baseline'
reform['Scenario'] = 'Reform Acceleration'
crisis['Scenario'] = 'External Crisis'
mixed['Scenario'] = 'Mixed Recovery'

# Combine forecast data
all_forecasts = pd.concat([baseline, reform, crisis, mixed], ignore_index=True)
forecast_df = pd.concat([baseline, reform, crisis, mixed], ignore_index=True)

# Ensure Year is integer in combined data
for frame in [all_forecasts, forecast_df]:
    frame["Year"] = frame["Year"].astype(int)

forecast_col = get_forecast_column(all_forecasts)

# Merge actual + SARIMAX predictions
df = df.merge(sarimax_df[["Year", "SARIMAX_Pred"]], on="Year", how="left")
df["Year"] = df["Year"].astype(int)

# ---- TITLE ----
st.title("India GDP Forecast(2025–2030)")
st.markdown("Visualizing scenario-based national GDP forecasts, risks, and strategic recommendations.")

# ---- HISTORICAL vs FINAL FORECAST PLOT ----
st.subheader("📉 Historical vs Forecasted GDP (1980–2030)")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df["Year"], df["GDP Growth (%)"], label="Actual GDP (1980–2024)", color='black', linewidth=2)

scenario_colors = {
    "Baseline": "green",
    "Reform Acceleration": "orange",
    "External Crisis": "red",
    "Mixed Recovery": "purple"
}

for scenario in forecast_df["Scenario"].dropna().unique():
    subset = forecast_df[forecast_df["Scenario"] == scenario]
    ax1.plot(
        subset["Year"],
        subset[forecast_col],
        label=f"{scenario} Forecast",
        linestyle='--',
        marker='o',
        color=scenario_colors.get(scenario, 'gray')
    )

ax1.set_xlabel("Year")
ax1.set_ylabel("GDP Growth (%)")
ax1.set_title("Actual vs Final Forecast (1980–2030)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1, use_container_width=True)

# ---- GDP GROWTH PROJECTION (2025–2030) ----
st.subheader("📈 GDP Growth Projections by Scenario (2025–2030)")
fig2, ax2 = plt.subplots(figsize=(8, 4))
for scenario, df_ in all_forecasts.groupby('Scenario'):
    ax2.plot(df_["Year"], df_[forecast_col], marker='o', label=scenario)

ax2.axhline(y=6.69, color='gray', linestyle='--', label='2025 Expected (6.69%)')
ax2.set_ylabel("GDP Growth (%)")
ax2.set_xlabel("Year")
ax2.legend()
st.pyplot(fig2, use_container_width=True)

# ---- BASELINE FORECAST DESCRIPTION ----
st.markdown("### 🟢 Baseline Forecast Description")
baseline_desc = baseline[["Year", forecast_col]].set_index("Year").T.to_dict()
desc_text = "The **baseline GDP growth forecast** is as follows:\n\n"
for year, value in baseline_desc.items():
    desc_text += f"- **{year}**: {value[forecast_col]:.2f}%\n"
st.markdown(desc_text)

# ---- SCENARIO COMPARISON TABLE (2027–2030) ----
st.subheader("📋 Scenario Forecast Comparison: 2027–2030 Only")
gdp_2027 = all_forecasts[
    (all_forecasts["Year"] >= 2027) &
    (all_forecasts["Scenario"] != 'Baseline')
]
table = gdp_2027.pivot(index='Year', columns='Scenario', values=forecast_col)
table.index = table.index.astype(int).astype(str)  # Remove decimal if any
st.dataframe(table.style.format("{:.2f}"), use_container_width=True)

# ---- ECONOMIC RECOMMENDATIONS ----
st.subheader("🧭 2025 Economic Recommendations")
try:
    with open("../../results/national/recommendations_2025.txt", "r", encoding="utf-8") as f:
        st.text(f.read())
except Exception as e:
    st.warning(f"Could not load recommendations file: {e}")

# ---- VISUAL INSIGHTS: ONLY CORRELATION HEATMAP ----
st.subheader("🔍 Feature Correlation Insight")
st.markdown("This heatmap visualizes the top 25 most correlated macroeconomic indicators used in the GDP forecasting model.")
st.image("../../results/national/plots/feature_correlation_top25.png", caption="Top 25 Most Correlated Macroeconomic Indicators", width=900)
