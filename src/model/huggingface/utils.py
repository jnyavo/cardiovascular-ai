from huggingface_hub import hf_hub_download
import os

def download_model(model_name: str, local_dir: str = None):
    """
    Download the model from Hugging Face Hub.
    Args:
        model_name (str): The name of the model to download.
    
    Returns:
        tuple: Paths to the downloaded model files (pt and yaml).
    """

    # Download the model

    local_dir = local_dir if local_dir else os.path.join(os.getcwd(), 'ckpts')

    basename = os.path.basename(model_name)

    dest = os.path.join(local_dir, basename)

    pt = hf_hub_download(
      repo_id=model_name,
      filename="physionet_finetuned.pt",
      local_dir=dest,
    )
    yaml = hf_hub_download(
        repo_id=model_name,
        filename='physionet_finetuned.yaml',
        local_dir=dest,
    )

    
    return (pt, yaml)