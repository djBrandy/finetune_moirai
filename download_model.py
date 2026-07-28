"""
download_model.py
-----------------
Downloads Salesforce/moirai-R-1.1-large locally to ./model_cache.
Skips download if already present.
"""

import os
from huggingface_hub import snapshot_download

MODEL_ID   = "Salesforce/moirai-1.1-R-large"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "model_cache")

def main():
    marker = os.path.join(LOCAL_PATH, "config.json")
    if os.path.exists(marker):
        print(f"Model already exists at {LOCAL_PATH}, skipping download.")
        return

    print(f"Downloading {MODEL_ID} to {LOCAL_PATH} ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_PATH,
        local_dir_use_symlinks=False,
    )
    print("Download complete.")

if __name__ == "__main__":
    main()
