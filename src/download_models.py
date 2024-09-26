import subprocess

def download_model(model_name: str, save_path: str):
    """Download a HistLLM from HuggingFace, if it is not already downloaded."""
    # huggingface-cli download PKU-Alignment/ProgressGym-HistLlama3-8B-C0${i}-instruct  --local-dir ./8B-C0${i}-instruct
    
    for _ in range(3): # Retry 3 times
        process = subprocess.Popen(
            ["huggingface-cli", "download", model_name, "--local-dir", save_path],
        )
        process.wait()
    
def download_all_models(download_8B = True, download_70B = False):
    """Download all HistLLMs from HuggingFace."""
    if download_8B:
        for i in range(13, 22):
            download_model(f"PKU-Alignment/ProgressGym-HistLlama3-8B-C0{i}-instruct", f"./dataset/dataset_model_sequence/8B-C0{i}-instruct")
    if download_70B:
        for i in range(13, 22):
            download_model(f"PKU-Alignment/ProgressGym-HistLlama3-70B-C0{i}-instruct", f"./dataset/dataset_model_sequence/70B-C0{i}-instruct")