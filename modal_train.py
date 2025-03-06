import modal
import os
import subprocess

# 1) Build a base image with your dependencies
#    Make sure you have a requirements.txt listing everything needed by train.py
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

# 2) Create a Volume for saving results (optional, but recommended)
results_volume = modal.Volume.from_name("fine_tuned_volume")

# 3) Define the Modal app (or 'stub')
stub = modal.Stub("DynaQ-RL-pipeline")

# 4) GPU + other config for the remote function
@stub.function(
    image=image,
    gpu="H100:1",            # or "H100:1", "V100:1", etc.
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
        "--lr=1e-3",
        "--batch_size=8",
    ]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Training complete!")


@stub.local_entrypoint()
def main():
    """
    Entry point when you run `modal run modal_train.py`.
    By default, runs a sample training job on the remote container.
    """
    run_train.remote(
        name="gpt-2-500-v0",
        model="gpt2",
        dataset="commonsense_qa",
        episodes=10,
    )
