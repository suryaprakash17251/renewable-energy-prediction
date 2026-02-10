import pandas as pd

df = pd.read_csv(r"C:\Users\chris\OneDrive\Desktop\renewable energy prediction\data\cleaned_energy_dataset.csv")
print("Unique state_name values:")
print(df['state_name'].unique()[:20])
