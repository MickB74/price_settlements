# VPPA Settlement Estimator

A Streamlit application for analyzing and comparing Virtual Power Purchase Agreement (VPPA) scenarios in the ERCOT market.

## Features

-   **Scenario Builder**: Create multiple scenarios with different parameters:
    -   Year (2023, 2024, 2025)
    -   Hub (North, South, West, Houston)
    -   Technology (Solar, Wind)
    -   Capacity (MW)
    -   Strike Price ($/MWh)
-   **Interactive Visualizations**:
    -   Cumulative Settlement ($) over time.
    -   Monthly Net Settlement ($).
    -   Monthly Energy Generation (MWh).
-   **Dynamic Insights**: Automatically generated insights highlighting the best/worst performing scenarios and months.
-   **Comparison**: Compare up to 10 distinct scenarios side-by-side.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/MickB74/price_settlements.git
    cd price_settlements
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Data Source

This application uses the [gridstatus](https://github.com/kmax12/gridstatus) library to fetch Real-Time Market (RTM) prices from ERCOT. Data is cached locally in `.parquet` format to improve performance on subsequent runs.
