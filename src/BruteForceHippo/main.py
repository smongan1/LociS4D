import os
import requests
# from train import train, CHECKPOINT_PATH
from train import CHECKPOINT_PATH_BEST, CHECKPOINT_PATH_LAST
from train_BruteForceHippo import main as train


def upload_model(checkpoint_path: str, put_url: str) -> None:
    """Upload the model checkpoint to S3 via a presigned PUT URL."""
    print(f"Uploading '{checkpoint_path}' to S3...")
    with open(checkpoint_path, "rb") as f:
        response = requests.put(
            put_url,
            data=f,
            headers={"Content-Type": "application/octet-stream"},
        )
    response.raise_for_status()
    print(f"Model uploaded successfully (HTTP {response.status_code}).")


def main():
    # ── Train ──────────────────────────────────────────────────────────────
    best_model = train()

    # ── Upload best checkpoint to S3 ───────────────────────────────────────
    model_put_best_url = os.environ.get("MODEL_PUT_BEST_URL")
    model_put_last_url = os.environ.get("MODEL_PUT_LAST_URL")
    if not model_put_best_url:
        print("Warning: MODEL_PUT_BEST_URL env var not set — skipping S3 upload.")
        
    if not model_put_last_url:
        print("Warning: MODEL_PUT_LAST_URL env var not set — skipping S3 upload.")

    if not os.path.exists(CHECKPOINT_PATH_BEST):
        print(f"Warning: checkpoint '{CHECKPOINT_PATH_BEST}' not found — skipping S3 upload.")
    else:
        upload_model(CHECKPOINT_PATH_BEST, model_put_best_url)
    if not os.path.exists(CHECKPOINT_PATH_LAST):
        print(f"Warning: checkpoint '{CHECKPOINT_PATH_LAST}' not found — skipping S3 upload.")
    else:
        upload_model(CHECKPOINT_PATH_LAST, model_put_last_url)

    return best_model

if __name__ == "__main__":
    main()
