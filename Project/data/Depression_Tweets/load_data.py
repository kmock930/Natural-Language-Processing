import kagglehub

# Download latest version
path = kagglehub.dataset_download("senapatirajesh/depression-tweets")

print("Path to dataset files:", path)