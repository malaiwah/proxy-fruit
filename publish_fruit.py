#!/usr/bin/env python3
"""Publish the Fruit pilot to HF."""
import os, sys
from huggingface_hub import HfApi

REPO = "malaiwah/GLM-5.2-SIQ-Fruit-pilot"
BASE = "/mnt/vault/llm/fruit-pilot"
SOURCE = os.path.dirname(os.path.abspath(__file__))
RUN_ASSETS = os.path.expanduser(
    os.environ.get("FRUIT_RUN_ASSETS", "~/fruit-pilot")
)

def main():
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
    print("uploading checkpoint (~0.48 GiB)...", flush=True)
    api.upload_folder(folder_path=f"{BASE}/output/GLM-5.2-SIQ-Fruit-pilot",
                      repo_id=REPO,
                      commit_message="GLM-5.2-SIQ-Fruit pilot: net-new micro "
                                     "GLM-5.2 mimic, trained then SIQ-encoded")
    api.upload_file(path_or_fileobj=f"{BASE}/fruit_pilot.pt",
                    path_in_repo="training/fruit_pilot.pt", repo_id=REPO,
                    commit_message="BF16 training state dict")
    for f in ("checkpoint_contract.py", "train_fruit.py", "export_fruit.py",
              "fruit_serve_test.py", "fruit_serve_r28.py"):
        api.upload_file(path_or_fileobj=f"{SOURCE}/{f}",
                        path_in_repo=f"tools/{f}", repo_id=REPO,
                        commit_message=f"tools: {f}")
    for f in ("train.log", "serve-r25.log", "serve-r28.log"):
        api.upload_file(path_or_fileobj=f"{RUN_ASSETS}/{f}",
                        path_in_repo=f"logs/{f}", repo_id=REPO,
                        commit_message=f"logs: {f}")
    api.upload_file(path_or_fileobj=f"{RUN_ASSETS}/modelcard/README.md",
                    path_in_repo="README.md", repo_id=REPO,
                    commit_message="model card")
    print(f"PUBLISHED: https://huggingface.co/{REPO}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
