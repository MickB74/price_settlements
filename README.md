# ERCOT VPPA Settlement Analyzer ⚡️

A professional-grade framework for modeling, analyzing, and comparing Virtual Power Purchase Agreement (VPPA) scenarios in the ERCOT market.

## 🌐 Live Application

**Access the app:** [price-settlements.streamlit.app](https://price-settlements.streamlit.app) *(or your actual Streamlit Cloud URL)*

## 🚀 Key Features

### 🛠️ Advanced Scenario Modeling (Beta)
-   **Multi-Year Analysis:** Simulate settlements for 2020–2026 using actual historical prices.
-   **Granular Location Data:**
    -   Select from major ERCOT Hubs.
    -   **Interactive Map Picker:** Pick any coordinate on an interactive map (now integrated seamlessly into the main workflow) or specify exact Lat/Lon.
-   **Weather-to-Power Profiles:**
    -   **2024–2026:** Uses **Actual Weather** (Open-Meteo ERA5 Reanalysis) for high-fidelity Solar & Wind profiles.
    -   **Historical (2005–2023):** Leverages **PVGIS** satellite-based solar and reanalysis wind data.
    -   **Multi-Weather Sensitivities:** Contrast "Actual Weather" vs. "Typical Meteorological Year" (TMY) profiles side-by-side to understand resource variability.
-   **Revenue Share Structures:** Model complex "Upside Sharing" PPAs (e.g., 50/50 splits when market price exceeds strike).
-   **Custom Profiles:** Support for 8760/15-min generation CSV uploads.
-   **🎲 Monte Carlo Simulation:** Generate probabilistic settlement outcomes (P10/P50/P90).
    -   Random sampling from 20 years of historical weather (2005-2024)
    -   Samples from 7 years of actual price data (2020-2026)
    -   Configurable iterations (100-10,000) to analyze risk profiles and outcome distributions

### 🏆 Bill Validation & Benchmarking (Beta)
-   **Bill Creation:** Create settlement bills based on weather models or validate incoming bills against official market prices.
-   **Curated Registry:** Select from **45+ major ERCOT assets** (Horse Hollow, Capricorn Ridge, Shaffer Wind, etc.) with pre-loaded metadata.
-   **Smart Defaults:**
    -   **Auto-Curtailment:** "Curtail when Price < $0" is enabled by default to reflect rational economic behavior.
    -   **Robust Location Handling:** Seamlessly switch between Hub-based and Custom Coordinate-based validation.
-   **Advanced Weather Context:**
    -   **Actual Weather:** Validate against what actually happened.
    -   **Typical Year (TMY):** Compare against long-term averages.
    -   **Calculated P50:** On-the-fly P50 calculation using 20 years of historical data.
    -   **Compare All:** Side-by-side view of Actual vs. TMY vs. P50 performance.
-   **SCED Validation:** Retrieve and compare model results against **real-world production data** from ERCOT's 60-day disclosure reports.
-   **High Correlation:** Model achieves an $R > 0.85$ correlation for top-tier coastal wind assets.

### 📊 Financial & Risk Analytics
-   **Settlement Engine:** 15-minute interval calculations of RTM vs. Strike Price.
-   **Financial Metrics:** Tracking of **Total Amount Paid**, **Total Amount Received**, and **Net Settlement**.
-   **Risk Modeling:** Visualize basis risk, negative pricing exposure, and the impact of economic curtailment ($0/MWh floor).
-   **Comparison Suite:** Side-by-side benchmarking of up to 10 distinct scenarios.
-   **Probabilistic Outcomes:** Monte Carlo distributions showing full range of possible results with percentile analysis.

### 📄 Professional Reporting
-   **PDF Settlement Bills:** Generate professional, client-ready PDF reports including monthly summaries and detailed daily performance metrics.
-   **Excel Exports:** Comprehensive data exports for external modeling and auditing.

## 📡 Data Sources

| Data Type | Source | Details |
| :--- | :--- | :--- |
| **Market Prices** | **ERCOT (via GridStatus)** | Real-Time Market (RTM) Hub & Load Zone prices (2020–2026). |
| **Actual Gen Data** | **ERCOT (SCED)** | 60-day delayed unit-level production for benchmarking. |
| **Solar/Wind (2024+)** | **Open-Meteo** | ERA5 Reanalysis (GHI, 10m Wind Speed) at 15-min resolution. |
| **Historical Weather**| **PVGIS** | Long-term satellite and reanalysis averages. |

## 📦 Installation & Setup

For local development or contributing to the project, see the [Installation Guide on GitHub](https://github.com/MickB74/price_settlements#installation).

## 📈 Validation Results
The framework's synthetic models have been validated against the actual ERCOT fleet. For a detailed breakdown of performance by region and technology, see the [Renewable Fleet Report](RENEWABLE_FLEET_REPORT.md).

## 💡 Usage Tips
-   **Map Search:** Use the "Pick Location" expander in the sidebar to search for any Texas town or project site.
-   **Benchmarking:** Target specific Resource IDs (e.g., `SHAFFER_UNIT1`) in the Bill Validation tab to fetch actual production data.
-   **Timezones:** All data is automatically synchronized to **Central Prevailing Time (CPT)** for market alignment.

## 📘 User Documentation
-   **User Guide + Workflow Chart:** See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for quick start steps, standard workflows, and a decision flowchart for end users.
