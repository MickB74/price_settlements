import pandas as pd
import numpy as np

def main():
    # Load benchmarking results
    try:
        df = pd.read_csv('solar_benchmarking_results.csv')
    except FileNotFoundError:
        print("Error: solar_benchmarking_results.csv not found.")
        return

    # Clean data (remove rows with missing actuals if any, though our script handled this)
    df = df[df['Annual Act'] > 0].copy()
    
    print(f"Analyzing {len(df)} projects...")

    # Current Logic: Estimate = Gross * 0.86 (14% losses)
    # So: Gross = Estimate / 0.86
    # We want: Gross * New_Efficiency = Actual
    # Therefore: New_Efficiency = Actual / Gross
    # New_Efficiency = Actual / (Estimate / 0.86)
    # New_Efficiency = (Actual * 0.86) / Estimate

    df['Implied_Efficiency'] = (df['Annual Act'] * 0.86) / df['Annual Est']
    df['Implied_Losses_Pct'] = (1 - df['Implied_Efficiency']) * 100

    # Stats
    mean_loss = df['Implied_Losses_Pct'].mean()
    median_loss = df['Implied_Losses_Pct'].median()
    std_loss = df['Implied_Losses_Pct'].std()
    
    # Weighted Average (by capacity)
    weighted_loss = np.average(df['Implied_Losses_Pct'], weights=df['Capacity (MW)'])

    print("\n--- Calibration Results ---")
    print(f"Current Loss Assumption: 14.00%")
    print(f"Mean Implied Loss:       {mean_loss:.2f}%")
    print(f"Median Implied Loss:     {median_loss:.2f}%")
    print(f"Weighted Implied Loss:   {weighted_loss:.2f}% (Weighted by Capacity)")
    print(f"Std Dev of Losses:       {std_loss:.2f}%")
    
    print("\n--- Recommendation ---")
    print(f"To center the error around 0%, change the default System Losses from 14% to approximately {median_loss:.0f}-{mean_loss:.0f}%.")
    
    # Save detailed calibration
    df[['Project Name', 'Capacity (MW)', 'Annual Est', 'Annual Act', 'Implied_Losses_Pct']].to_csv('solar_calibration_analysis.csv', index=False)
    print("Detailed analysis saved to solar_calibration_analysis.csv")

if __name__ == "__main__":
    main()
