import pandas as pd

# Load the Excel file
df = pd.read_excel('dataset_gabungan.xlsx', sheet_name='Sheet1')

# Save to CSV
df.to_csv('dataset_gabungan.csv', index=False)
