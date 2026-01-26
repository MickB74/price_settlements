import pandas as pd
import os
import glob
import sys

def main():
    # 1. Load Estimates
    est_file = 'solar_monthly_output_2025.csv'
    if not os.path.exists(est_file):
        print(f"Error: {est_file} not found. Run generate_solar_monthly.py first.")
        return

    print(f"Loading estimates from {est_file}...")
    est_df = pd.read_csv(est_file)
    
    # 2. Setup Test Data Search
    test_data_dir = 'test data'
    if not os.path.exists(test_data_dir):
        print(f"Error: Directory '{test_data_dir}' not found.")
        return

    results = []
    month_map = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 
        5: 'May', 6: 'June', 7: 'July', 8: 'August', 
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }

    print(f"Benchmarking {len(est_df)} projects...")

    for index, row in est_df.iterrows():
        project_name = row['Project Name']
        project_id = row['Project ID']
        
        # Handle NaN Project ID
        if pd.isna(project_id):
            print(f"Skipping {project_name}: Missing Project ID")
            continue
            
        project_id_str = str(int(project_id)) # Convert to string, remove decimals
        
        # 3. Find Matching Test File
        # Pattern: *<ProjectID>.csv
        search_pattern = os.path.join(test_data_dir, f"*{project_id_str}.csv")
        matches = glob.glob(search_pattern)
        
        if not matches:
            print(f"Warning: No test file found for {project_name} (ID: {project_id_str})")
            continue
            
        test_file = matches[0]
        # print(f"Comparing {project_name} vs {os.path.basename(test_file)}...")
        
        try:
            # 4. Load and Aggregate Test Data
            # Handle potential 'sep=,' line
            try:
                # Try reading normally first
                actual_df = pd.read_csv(test_file)
                if 'Month' not in actual_df.columns:
                     # If typical columns missing, maybe first line is garbage
                     actual_df = pd.read_csv(test_file, skiprows=1)
            except Exception:
                 actual_df = pd.read_csv(test_file, skiprows=1)
            
            # Check columns
            if 'Month' not in actual_df.columns or 'PRODUCTION OUTPUT in MWh at 100% availability' not in actual_df.columns:
                print(f"Error: Unexpected columns in {test_file}")
                continue
                
            # Aggregate by Month
            actual_monthly = actual_df.groupby('Month')['PRODUCTION OUTPUT in MWh at 100% availability'].sum()
            
            project_result = {
                'Project Name': project_name,
                'Project ID': project_id_str,
                'Capacity (MW)': row['Capacity (MW)']
            }
            
            total_est = 0
            total_act = 0
            
            # 5. Compare Monthly
            for month_num in range(1, 13):
                month_name = month_map[month_num]
                
                # Estimated MWh (Field name is Month Name)
                est_mwh = row.get(month_name, 0)
                
                # Actual MWh
                act_mwh = actual_monthly.get(month_num, 0)
                
                # Calc Diff
                diff = est_mwh - act_mwh
                pct_err = (diff / act_mwh * 100) if act_mwh > 0 else 0
                
                project_result[f'{month_name} Est'] = est_mwh
                project_result[f'{month_name} Act'] = act_mwh
                project_result[f'{month_name} Diff'] = diff
                project_result[f'{month_name} Err %'] = pct_err
                
                total_est += est_mwh
                total_act += act_mwh
            
            # Annual Stats
            project_result['Annual Est'] = total_est
            project_result['Annual Act'] = total_act
            project_result['Annual Diff'] = total_est - total_act
            project_result['Annual Err %'] = ((total_est - total_act) / total_act * 100) if total_act > 0 else 0
            
            results.append(project_result)
            
        except Exception as e:
            print(f"Error processing {test_file}: {e}")

    # 6. Save Results
    if results:
        bench_df = pd.read_json(pd.DataFrame(results).to_json()) # normalize
        output_file = 'solar_benchmarking_results.csv'
        bench_df.to_csv(output_file, index=False)
        print(f"\nBenchmarking Complete.")
        print(f"Processed {len(results)} projects.")
        print(f"Saved details to {output_file}")
        
        # Summary Prints
        avg_err = bench_df['Annual Err %'].mean()
        print(f"Average Annual Error: {avg_err:.2f}%")
        
    else:
        print("No benchmarking results generated.")

if __name__ == "__main__":
    main()
