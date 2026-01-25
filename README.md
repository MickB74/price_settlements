# ERCOT VPPA Settlement Analyzer ⚡️

A professional-grade framework for modeling, analyzing, and comparing Virtual Power Purchase Agreement (VPPA) scenarios in the ERCOT market.

## 🚀 Key Features

### 🛠️ Advanced Scenario Modeling
-   **Multi-Year Analysis:** Simulate settlements for 2020–2026 using actual historical prices.
-   **Granular Location Data:** Select from major ERCOT Hubs or pick any coordinate on an interactive map.
-   **Weather-to-Power Profiles:**
    -   **2024–2026:** Uses **Actual Weather** (Open-Meteo ERA5 Reanalysis) for high-fidelity Solar & Wind profiles.
    -   **Historical (2005–2023):** Leverages **PVGIS** satellite-based solar and reanalysis wind data.
    -   **Multi-Weather Sensitivities:** Contrast "Actual Weather" vs. "Typical Meteorological Year" (TMY) profiles side-by-side to understand resource variability.
-   **Revenue Share Structures:** Model complex "Upside Sharing" PPAs (e.g., 50/50 splits when market price exceeds strike).
-   **Custom Profiles:** Support for 8760/15-min generation CSV uploads.

### 🏆 Real Asset Benchmarking
-   **Curated Registry:** Select from **45+ major ERCOT assets** (Horse Hollow, Capricorn Ridge, Shaffer Wind, etc.) with pre-loaded metadata.
-   **Interactive Map Picker:** Click anywhere in Texas to auto-find the nearest settlement hub or specify exact coordinates via geocoding (search by town/ZIP).
-   **SCED Validation:** Retrieve and compare model results against **real-world production data** from ERCOT's 60-day disclosure reports.
-   **High Correlation:** Model achieves an $R > 0.85$ correlation for top-tier coastal wind assets.

### 📊 Financial & Risk Analytics
-   **Settlement Engine:** 15-minute interval calculations of RTM vs. Strike Price.
-   **Financial Metrics:** Tracking of **Total Amount Paid**, **Total Amount Received**, and **Net Settlement**.
-   **Risk Modeling:** Visualize basis risk, negative pricing exposure, and the impact of economic curtailment ($0/MWh floor).
-   **Comparison Suite:** Side-by-side benchmarking of up to 10 distinct scenarios.

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

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MickB74/price_settlements.git
    cd price_settlements
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## 📈 Validation Results
The framework's synthetic models have been validated against the actual ERCOT fleet. For a detailed breakdown of performance by region and technology, see the [Renewable Fleet Report](RENEWABLE_FLEET_REPORT.md).

## 💡 Usage Tips
-   **Map Search:** Use the "Pick Location" expander in the sidebar to search for any Texas town or project site.
-   **Benchmarking:** Target specific Resource IDs (e.g., `SHAFFER_UNIT1`) in the Bill Validation tab to fetch actual production data.
-   **Timezones:** All data is automatically synchronized to **Central Prevailing Time (CPT)** for market alignment.
