import pandas as pd
import os

# Load the CSV file into a DataFrame
FILEPATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "sentimentdataset.csv")
df = pd.read_csv(FILEPATH)

# Display the first few rows of the DataFrame
print(df.head())

# Display all features
print("Columns:\n", df.columns)

# Display the shape of the DataFrame
print("Shape: ", df.shape)