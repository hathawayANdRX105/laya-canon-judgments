"""Headless laya RLCD training on Modal.

Free Starter plan: $30/month credits ~= 50 T4-h (Modal's own
pricing table). Script-shaped: no notebook wrapper needed.

Setup (once):
    pip install modal
    modal setup
    modal secret create laya-hf HF_TOKEN=<hf token>  # checkpoint push

Run:
    modal run cloud/modal_train.py
    modal run cloud/modal_train.py --epochs 8 --unfreeze

Checkpoint is pushed to HF Hub (HF_REPO below) so it survives
the ephemeral container; the train report is returned and printed.

GPU is fixed at decoration time: default T4 (16GB) fits the
head-only stage; switch `gpu=` to "A10G" (24GB) for unfrozen
encoder runs. Data (splits) is cloned from the judgments repo.
"""
import argparse

import modal

LAYA_GIT = "https://github.com/NandhaKishorM/laya"
JUDGMENTS_GIT = "https://github.com/hathawayANdRX105/laya-canon-judgments"
HF_REPO = "hathawayandrx105/laya-canon-head"

app = modal.App("laya-rlcd")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel")
    .apt_install("git")
    .pip_install("transformers", "datasets", "huggingface_hub", "accelerate")
    .run_commands(f"pip install -e git+{LAYA_GIT}")
)


@app.function(
    gpu="T4",
    image=image,
    secrets=[modal.Secret.from_name("laya-hf")],
    timeout=3600,
)
def train(epochs: int = 4, unfreeze: bool = False):
    import os
    import subprocess

    subprocess.run(
        ["git", "clone", "--depth", "1", JUDGMENTS_GIT, "/work/j"], check=True
    )
    from laya.train import TrainConfig, finetune
    from laya.agent import _fix_tokenizer_config

    model_dir = "convaiinnovations/laya"
    _fix_tokenizer_config(model_dir)
    config = TrainConfig(
        epochs=epochs,
        micro_batch=2,
        grad_accum=4,
        freeze_encoder=not unfreeze,
        seed=42,
        calib_seed=42,
        max_len=1024,
        head_max_len=256,
    )
    out = "/work/out"
    summary = finetune(
        data="/work/j/data/splits/train.jsonl",
        model_dir=model_dir,
        output_dir=out,
        config=config,
        device="cuda",
    )
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(HF_REPO, exist_ok=True)
    api.upload_folder(repo_id=HF_REPO, folder_path=out, path_in_repo="")
    return {"summary": str(summary)}


@app.local_entrypoint()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--unfreeze", action="store_true",
                   help="unfreeze the encoder (needs A10G: edit gpu= above)")
    a = p.parse_args()
    train.remote(a.epochs, a.unfreeze)
