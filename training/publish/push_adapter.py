from huggingface_hub import HfApi


def push_adapter(repo_id: str, folder_path: str, private: bool = False):
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api.upload_folder(repo_id=repo_id, folder_path=folder_path)
