import json
import os

# Path to the JSON file
FILENAME = 'reddit_suicidewatch.json'
PATH = os.path.join(os.path.dirname(__file__), FILENAME)

# Load the JSON file
with open(PATH, 'r') as file:
    data = json.load(file)

# Print the properties of the JSON data
print("Properties in JSON with their datatypes: ")
for key, value in data.items():
    print(f"{key}: {type(value)}")

# Print the KIND property
print("KIND property:\n", data['kind'])

# Print the DATA property
print("DATA property attributes:")
for key, value in data['data'].items():
    print(f"{key}: {type(value)}")

# Print the CHILDREN property
print("CHILDREN property:")
print("Number of children:", len(data['data']['children']))
print("First child:")
for key, value in data['data']['children'][0].items():
    print(f"{key}: {type(value)}")

# Print the DATA property of the first child
print("DATA property of the first child:")
for key, value in data['data']['children'][0]['data'].items():
    print(f"{key}: {type(value)}")

# Output the first child into a JSON file
with open(os.path.join(os.path.dirname(__file__), 'reddit_posts_data_first_child.json'), 'w') as file:
    json.dump(data['data']['children'][0], file, indent=4)