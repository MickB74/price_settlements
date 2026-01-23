# ERCOT VPPA Settlement Analyzer ⚡️

A professional-grade framework for modeling, analyzing, and comparing Virtual Power Purchase Agreement (VPPA) scenarios in the ERCOT market.

## 🚀 Key Features

### 🛠️ Scenario Modeling
-   **Multi-Year Analysis:** Simulate settlements for 2020-2026.
-   **Granular Location Data:** Select from all major ERCOT Hubs (North, South, West, Houston, Panhandle) or pick any coordinate on an interactive map.
-   **Weather-Driven Profiles:**
    -   **2024-2026:** Uses **Actual Weather** (Open-Meteo ERA5) for both Solar & Wind.
    -   **History (2005-2023):** Uses **Actual Weather** (PVGIS).
    -   **Sensitivity:** Toggle "Force TMY" to compare actuals against "Typical Meteorological Year" baselines.
-   **Revenue Share Modeling:** Toggle revenue share options (e.g., 50/50 split) for more complex PPA settlement structures where buyer only gets a percentage of the upside when SPP exceeds the strike price.
-   **Custom Profiles:** Upload your own 8760/15-min generation shapes (CSV) for bespoke analysis.

### 📊 Financial Analysis
-   **Settlement Engine:** Calculates RTM vs. Strike Price settlements at 15-minute intervals.
-   **Bill Validation:** Compare actual market revenue against synthetic settlements for specific months and locations to validate model accuracy.
-   **Risk Metrics:** 
    -   Visualize cumulative P&L.
    -   Analyze basis risk and negative pricing exposure.
    -   Curtailment modeling (optional economic curtailment at $0/MWh).
-   **Comparison:** Side-by-side comparison of up to 10 distinct scenarios.

### 📈 Interactive Visualization
-   Net Settlement heatmaps.
-   Monthly financial performance aggregation.
-   Generation vs. Price correlation plots.

## 📡 Data Sources

| Data Type | Source | Details |
| :--- | :--- | :--- |
| **Market Prices** | **ERCOT (via GridStatus)** | Real-Time Market (RTM) settlement point prices. |
| **Solar/Wind (2024-2026)** | **Open-Meteo** | ERA5 Reanalysis data (Global Horizontal Irradiance, 10m Wind Speed). |
| **Solar/Wind (History)** | **PVGIS** | Satellite-based solar radiation and reanalysis wind data. |
| **TMY Data** | **PVGIS** | Typical Meteorological Year derived from multi-year averages. |

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

## 💡 Usage Tips
-   **Caching:** The app locally caches weather and pricing data (`/data_cache`) to speed up subsequent runs.
-   **Custom Uploads:** Prepare a CSV with a `Gen_MW` column (and optional `Time` column) to analyze specific project shapes.
