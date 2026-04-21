from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="model.pt",
    path_in_repo="model.pt",
    repo_id="BinLi0206/InterAgent",
    repo_type="model",
)