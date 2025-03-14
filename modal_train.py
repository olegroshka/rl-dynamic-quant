import os
import pathlib
import subprocess
from time import sleep

import modal

# 1) Build a base image with your dependencies
#    Make sure you have a requirements.txt listing everything needed by train.py
image = (modal.Image.debian_slim().pip_install_from_requirements("requirements.txt"))

# 2) Create a Volume for saving results (optional, but recommended)
results_volume = modal.Volume.from_name("fine_tuned_volume")

# 3) Define the Modal app (or 'stub')
stub = modal.App("DynaQ-RL-pipeline")

# 4) GPU + other config for the remote function
@stub.function(
    image=image,
    gpu="H100:1",            # or "H100:1", "V100:1", etc.
    block_network=False,
    timeout=86400,
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
        modal.Secret.from_name("dynaq-wandb-secret"),
    ],
    # Mount your local code so train.py is available in /root/code
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/code")],
    # Attach a persistent volume for saving any outputs
    volumes={"/root/results": results_volume},
)
def run_train(
    name: str = "test-run",
    model: str = "gpt2",
    dataset: str = "commonsense_qa",
    episodes: int = 10,
):
    # remove any local GPT2 cache that might overshadow the Hugging Face model name

    """
    Calls train.py with the desired arguments on a remote GPU container.
    """
    # Ensure the WANDB_API_KEY is set from the secret
    os.environ["WANDB_API_KEY"] = os.environ.get("WANDB_API_KEY", "")

    # Build up the command to run train.py
    # You can add or remove any CLI arguments you need here.
    cmd = [
        "python",
        "/root/code/train.py",  # Path to train.py inside the container
        f"--name={name}",
        f"--model={model}",
        f"--dataset={dataset}",
        f"--episodes={episodes}",
        "--finetune_steps=30",
        "--lr=1e-3",
        "--batch_size=32",
    ]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Training complete!")


@stub.local_entrypoint()
def main():
    """
    When run locally (`modal run modal_train.py`), this will:
      1) Kick off training remotely (run_train).
      2) After training is done, download the results from the volume to a local folder.
    """
    # 1) Start the remote training run
    experiment_name = "gpt2-250-gae-sig-ewa-rwd-v6"    # The name passed above

    run_train.remote(
        name=experiment_name,
        model="gpt2",
        dataset="commonsense_qa",
        episodes=250,
    )

    sleep(5)

    # 3) Use Modal CLI to download results automatically
    local_dir = f"results/"
    volume_remote_path = f"/{experiment_name}"  # Note: root ('/') of volume is directly referenced by Modal CLI

    # Ensure local directory exists
    pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)

    # Modal CLI command to get files from the remote volume
    subprocess.run([
        "modal", "volume", "get",
        "fine_tuned_volume",
        volume_remote_path,
        local_dir
    ], check=True)

    print(f"Results for {experiment_name} have been downloaded to {local_dir}/{experiment_name}")