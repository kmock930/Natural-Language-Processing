import json
import os

# Path to the JSON file
FILENAME = 'depression_json'
PATH = os.path.join(os.path.dirname(__file__), FILENAME)

# Load the JSON file
with open(PATH, 'r') as file:
    data = json.load(file)

# Print the properties of the JSON data
print("Properties in JSON with their datatypes: ")
for key, value in data.items():
    print(f"{key}: {type(value)}")

# Print the properties in dictionary
print("\nProperties in JSON: ")
for key in data.keys():
    print(f"{key}: {type(data[key])}")

# Observe CONTENT property
print("\nCONTENT property: ")
print("Length of sample data: ", len(data['content']))
print("Type of data: ", type(data['content']))

# First 5 elements of CONTENT property
for content_key in list(data['content'].keys())[:5]:
    first_5_elements = data['content'][content_key]

# Save into a new JSON file
output_path = os.path.join(os.path.dirname(__file__), 'depression_tweets_first_five_elements.json')
with open(output_path, 'w') as output_file:
    json.dump(first_5_elements, output_file)
    print(f"First 5 elements saved to {output_path}")