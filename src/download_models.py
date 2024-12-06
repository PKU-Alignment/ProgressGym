from src.path import root
import subprocess, os, sys


def download_model(model_name: str, save_path: str):
    """Download a model from HuggingFace, if it is not already downloaded."""
    for _ in range(3):  # Retry 3 times
        with open(os.devnull, "w") as devnull:
            process = subprocess.Popen(
                ["huggingface-cli", "download", model_name, "--local-dir", save_path],
                stdout=(devnull if not eval(os.environ.get("LOUD_BACKEND", "False")) else sys.stdout),
                stderr=(devnull if not eval(os.environ.get("LOUD_BACKEND", "False")) else sys.stderr),
            )
            process.wait()


def download_all_models(download_8B=True, download_70B=False):
    """Download all HistLLMs from HuggingFace."""
    if download_8B:
        for i in range(13, 22):
            download_model(
                f"PKU-Alignment/ProgressGym-HistLlama3-8B-C0{i}-pretrain",
                f"{root}/dataset/dataset_model_sequence/8B-C0{i}-pretrain",
            )
    if download_70B:
        for i in range(13, 22):
            download_model(
                f"PKU-Alignment/ProgressGym-HistLlama3-70B-C0{i}-instruct",
                f"{root}/dataset/dataset_model_sequence/70B-C0{i}-instruct",
            )
