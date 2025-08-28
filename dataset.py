import kagglehub

# Download latest version
path = kagglehub.dataset_download("imdeepmind/preprocessed-trec-2007-public-corpus-dataset")

print("Path to dataset files:", path)