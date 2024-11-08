import pandas as pd
import numpy as np
import calendar
from openpyxl import Workbook

# Load data
file_path = 'bed-occupancy-rate_week40y2024.xlsx'
excel_data = pd.ExcelFile(file_path)
historical_data = excel_data.parse('BOR(%)_historical')

# Format the data
historical_data.columns = historical_data.iloc[1]  # Use second row as header
historical_data = historical_data.drop([0, 1]).reset_index(drop=True)  # Drop initial rows

# Convert 'Date' column to datetime format
historical_data['Date'] = pd.to_datetime(historical_data['Date'], errors='coerce')

# Define years, months, and hospital columns
years = historical_data['Years'].unique()
months = historical_data['Date'].dt.month.unique()
hospitals = historical_data.columns[2:]  # Assuming hospitals start from the 3rd column

# Initialize a DataFrame to hold the consolidated data
consolidated_data = pd.DataFrame()

for hospital in hospitals:
    # Initialize a dictionary to store simulated averages for each day
    hospital_data = {}

    for month in months:
        # Filter data for the specific month across all years for this hospital
        monthly_data = historical_data[historical_data['Date'].dt.month == month][['Date', 'Years', hospital]].copy()
        
        # Check if data is empty for the month
        if monthly_data.empty:
            continue

        # Add a day column for easier grouping and analysis
        monthly_data['Day'] = monthly_data['Date'].dt.day

        # Pivot the data to have years as columns and days as rows
        monthly_pivot = monthly_data.pivot(index='Day', columns='Years', values=hospital)

        # Calculate min and max for each day across years
        monthly_pivot['Min'] = monthly_pivot.min(axis=1)
        monthly_pivot['Max'] = monthly_pivot.max(axis=1)

        # Simulate 1000x for each day and calculate the average of these simulations
        def simulate_avg(row):
            # Handle case where both min and max are NaN (no data for that day)
            if pd.isna(row['Min']) or pd.isna(row['Max']):
                return np.nan
            simulations = np.random.uniform(row['Min'], row['Max'], 1000)
            return simulations.mean()
        
        monthly_pivot['Simulated Avg'] = monthly_pivot.apply(simulate_avg, axis=1)

        # Add simulated average to hospital_data with date format as '1 Jan', '2 Jan', etc.
        month_name = calendar.month_name[month]
        for day, avg in monthly_pivot['Simulated Avg'].items():
            date_key = f"{day} {month_name}"
            hospital_data[date_key] = avg

    # Convert hospital data to a DataFrame and add to consolidated data
    hospital_df = pd.DataFrame.from_dict(hospital_data, orient='index', columns=[hospital])
    consolidated_data = pd.concat([consolidated_data, hospital_df], axis=1)

# Save consolidated data to an Excel file
output_file = "hospital_occupancy_simulation_consolidated.xlsx"
consolidated_data.index.name = "Date"  # Set index name for clarity
consolidated_data.to_excel(output_file)
print(f"Data has been saved to {output_file}")
