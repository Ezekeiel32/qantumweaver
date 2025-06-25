import os
import subprocess
from typing import Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
try:
    import git
except ImportError:
    git = None

def download_kaggle_dataset(dataset: str, dest_dir: str, kaggle_username: Optional[str] = None, kaggle_key: Optional[str] = None):
    os.makedirs(dest_dir, exist_ok=True)
    if kaggle_username and kaggle_key:
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_key
    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", dest_dir, "--unzip"]
        subprocess.run(cmd, check=True)
        return {"status": "success", "message": f"Kaggle dataset {dataset} downloaded to {dest_dir}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def download_hf_dataset(dataset_name: str, dest_dir: str, split: str = "train"):
    if load_dataset is None:
        return {"status": "error", "message": "Hugging Face 'datasets' package not installed."}
    os.makedirs(dest_dir, exist_ok=True)
    try:
        ds = load_dataset(dataset_name, split=split)
        ds.save_to_disk(dest_dir)
        return {"status": "success", "message": f"HF dataset {dataset_name} saved to {dest_dir}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def download_github_repo(repo_url: str, dest_dir: str):
    if git is None:
        return {"status": "error", "message": "'gitpython' package not installed."}
    os.makedirs(dest_dir, exist_ok=True)
    try:
        git.Repo.clone_from(repo_url, dest_dir)
        return {"status": "success", "message": f"GitHub repo {repo_url} cloned to {dest_dir}"}
    except Exception as e:
        return {"status": "error", "message": str(e)} 