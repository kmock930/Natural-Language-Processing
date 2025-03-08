import kagglehub

# Download latest version
path = kagglehub.dataset_download("hosammhmdali/twitter-suicidal-data")

print("Path to dataset files:", path)