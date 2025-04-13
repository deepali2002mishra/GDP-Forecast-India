import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === PAGE SETUP ===
st.set_page_config(page_title="India GDP Forecast Dashboard (National & Sectoral)", layout="wide")
st.title("India GDP Forecast Dashboard (2025–2030)")
st.markdown("Explore national GDP projections and sector-specific forecasts across Agriculture and IT.")

# === BASE PATH SETUP ===
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results", "national")
data_dir = os.path.join(base_dir, "data", "processed")

# === MAIN TABS ===
main_tab1, main_tab2 = st.tabs(["📊 National GDP", "📂 Sectoral GDP"])

# === NATIONAL GDP TAB ===
with main_tab1:
    st.header("National GDP Forecast (1980–2030)")

    def clean_columns(df):
        df.columns = df.columns.str.strip()
        return df

    def get_forecast_column(df):
        forecast_cols = [col for col in df.columns if "forecast" in col.lower()]
        if forecast_cols:
            return forecast_cols[0]
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        return numeric_cols[-1]

    baseline = clean_columns(pd.read_csv(os.path.join(results_dir, "gdp_forecast_baseline_2025_2026.csv")))
    reform = clean_columns(pd.read_csv(os.path.join(results_dir, "gdp_forecast_reform_2027_2030.csv")))
    crisis = clean_columns(pd.read_csv(os.path.join(results_dir, "gdp_forecast_crisis_2027_2030.csv")))
    mixed = clean_columns(pd.read_csv(os.path.join(results_dir, "gdp_forecast_mixed_2027_2030.csv")))
    sarimax_df = pd.read_csv(os.path.join(data_dir, "sarimax_predictions.csv"))
    df = pd.read_csv(os.path.join(data_dir, "processed_data.csv"))

    for frame in [baseline, reform, crisis, mixed, sarimax_df, df]:
        frame["Year"] = frame["Year"].astype(int)

    baseline['Scenario'] = 'Baseline'
    reform['Scenario'] = 'Reform Acceleration'
    crisis['Scenario'] = 'External Crisis'
    mixed['Scenario'] = 'Mixed Recovery'

    forecast_df = pd.concat([baseline, reform, crisis, mixed], ignore_index=True)
    forecast_col = get_forecast_column(forecast_df)
    df = df.merge(sarimax_df[["Year", "SARIMAX_Pred"]], on="Year", how="left")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("📉 Historical vs Forecasted GDP (1980–2030)")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(df["Year"], df["GDP Growth (%)"], label="Actual GDP (1980–2024)", color='black', linewidth=2)
        scenario_colors = {
            "Baseline": "green",
            "Reform Acceleration": "orange",
            "External Crisis": "red",
            "Mixed Recovery": "purple"
        }
        for scenario in forecast_df["Scenario"].dropna().unique():
            subset = forecast_df[forecast_df["Scenario"] == scenario]
            ax1.plot(subset["Year"], subset[forecast_col], label=f"{scenario} Forecast", linestyle='--', marker='o', color=scenario_colors.get(scenario, 'gray'))
        ax1.set_xlabel("Year")
        ax1.set_ylabel("GDP Growth (%)")
        ax1.set_title("Actual vs Forecast")
        ax1.grid(True)
        ax1.legend(fontsize=6)
        st.pyplot(fig1)

    with col2:
        st.subheader("📈 GDP Growth Projections by Scenario (2025–2030)")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        for scenario, df_ in forecast_df.groupby('Scenario'):
            ax2.plot(df_["Year"], df_[forecast_col], marker='o', label=scenario)
        ax2.axhline(y=6.69, color='gray', linestyle='--', label='2025 Expected (6.69%)')
        ax2.set_ylabel("GDP Growth (%)")
        ax2.set_xlabel("Year")
        ax2.set_title("Scenario-Wise Projections")
        ax2.legend(fontsize=6)
        st.pyplot(fig2)

    st.markdown("### 🟢 Baseline Forecast Description")
    baseline_desc = baseline[["Year", forecast_col]].set_index("Year").T.to_dict()
    desc_text = "The **baseline GDP growth forecast** is as follows:\n\n"
    for year, value in baseline_desc.items():
        desc_text += f"- **{year}**: {value[forecast_col]:.2f}%\n"
    st.markdown(desc_text)

    st.subheader("📋 Scenario Forecast Comparison: 2027–2030 Only")
    gdp_2027 = forecast_df[(forecast_df["Year"] >= 2027) & (forecast_df["Scenario"] != 'Baseline')]
    table = gdp_2027.pivot(index='Year', columns='Scenario', values=forecast_col)
    table.index = table.index.astype(int).astype(str)
    st.dataframe(table.style.format("{:.2f}"), use_container_width=True)

    st.subheader("🧭 2025 Economic Recommendations")
    rec_path = os.path.join(results_dir, "recommendations_2025.txt")
    if os.path.exists(rec_path):
        with open(rec_path, "r", encoding="utf-8") as f:
            st.text(f.read())
    else:
        st.warning("⚠️ Recommendations file not found.")

# === SECTORAL GDP TAB ===
with main_tab2:
    sector = st.radio("Choose a sector:", ["🌾 Agriculture", "💻 IT Sector"], horizontal=True)

    if sector == "🌾 Agriculture":
        st.header("Agricultural Sector Forecasts & Investment Analysis")
        agri_states = ["Andhra_Pradesh", "Assam", "Gujarat", "Haryana", "Karnataka",
                       "Madhya_Pradesh", "Maharashtra", "Punjab", "Uttar_Pradesh", "West_Bengal"]
        selected = st.selectbox("Select a State", agri_states, format_func=lambda x: x.replace('_', ' '))

        agri_plot_dir = os.path.join(base_dir, "results", "sectoral", "agriculture", "plots")
        agri_report_dir = os.path.join(base_dir, "results", "sectoral", "agriculture", "reports")
        matched_plot = next((os.path.join(agri_plot_dir, f) for f in os.listdir(agri_plot_dir)
                             if f.startswith(selected) and f.endswith(".png")), None)
        report_path = os.path.join(agri_report_dir, f"{selected}_report.txt")

        col1, col2 = st.columns([0.65, 0.35], gap="large")
        with col1:
            if matched_plot and os.path.exists(matched_plot):
                st.image(matched_plot, caption=f"{selected.replace('_', ' ')} – Forecast", use_container_width=True)
            else:
                st.warning(f"⚠️ Forecast image not found for {selected.replace('_', ' ')}")

        with col2:
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as file:
                    st.text(file.read())
            else:
                st.warning(f"⚠️ Report not found for {selected.replace('_', ' ')}")

    elif sector == "💻 IT Sector":
        st.header("IT Sector Forecast & Strategic Recommendations")

        it_plot_dir = os.path.join(base_dir, "results", "sectoral", "IT", "plots")
        it_report_dir = os.path.join(base_dir, "results", "sectoral", "IT", "reports")
        forecast_img = os.path.join(it_plot_dir, "all_states_forecast_IT.png")
        top3_img = os.path.join(it_plot_dir, "top3_growth_IT.png")
        report_path = os.path.join(it_report_dir, "it_sector_top3_report.txt")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("📈 State-wise IT Revenue Forecast (2021–2030)")
            if os.path.exists(forecast_img):
                st.image(forecast_img, caption="State-wise IT Revenue Forecast", use_container_width=True)
            else:
                st.warning("⚠️ IT forecast plot not found.")

        with col2:
            st.subheader("🚀 Top 3 States by Projected Growth Rate")
            if os.path.exists(top3_img):
                st.image(top3_img, caption="Top 3 States by Projected Growth", use_container_width=True)
            else:
                st.warning("⚠️ Top 3 growth plot not found.")

        st.subheader("📋 Strategic Investment Recommendations")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                st.text(f.read())
        else:
            st.warning("⚠️ IT sector strategy report not found.")
