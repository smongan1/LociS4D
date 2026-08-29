import json
import os
import boto3
import argparse
from botocore.config import Config

def parse_arguments():
    """Parses command-line arguments and merges them with JSON defaults."""
    parser = argparse.ArgumentParser(
        description="Upload a training script to RunPod Network Storage using S3 API."
    )
    
    # Configuration File
    parser.add_argument(
        "--config", type=str, default="runpod_config.json",
        help="Path to the JSON configuration file (default: runpod_config.json)"
    )
    
    # Infrastructure Overrides
    parser.add_argument(
        "--volume_id", type=str, 
        help="RunPod Network Volume ID (overrides JSON)"
    )
    parser.add_argument(
        "--datacenter", type=str, 
        help="RunPod Datacenter ID, e.g., US-SFC-1 (overrides JSON)"
    )
    
    # File Path Overrides
    parser.add_argument(
        "--local_script", type=str, 
        help="Path to the local Python script to upload (overrides JSON)"
    )
    parser.add_argument(
        "--remote_key", type=str, 
        help="Remote destination path inside /workspace (overrides JSON)"
    )

    args = parser.parse_args()

    # 1. Load defaults from JSON if it exists
    config_data = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_data = json.load(f)
    else:
        print(f"⚠️ Warning: Config file '{args.config}' not found. Relying solely on CLI flags.")

    # 2. Build the final unified configuration (CLI arguments override JSON)
    infra_cfg = config_data.get("infrastructure", {})
    paths_cfg = config_data.get("deployment_paths", {})
    auth_cfg = config_data.get("runpod_auth", {})

    final_settings = {
        # Auth environment variable mappings
        "api_key_var": auth_cfg.get("api_key_env_var", "RUNPOD_API_KEY"),
        "s3_access_var": auth_cfg.get("s3_access_key_env_var", "RUNPOD_S3_ACCESS_KEY"),
        "s3_secret_var": auth_cfg.get("s3_secret_key_env_var", "RUNPOD_S3_SECRET_KEY"),
        
        # Core configurations (CLI flag || JSON || None)
        "network_volume_id": args.volume_id or infra_cfg.get("network_volume_id"),
        "datacenter_id": args.datacenter or infra_cfg.get("datacenter_id"),
        "local_file_path": args.local_script or paths_cfg.get("local_script_path"),
        "remote_s3_key": args.remote_key or paths_cfg.get("remote_script_key")
    }

    # 3. Validation safeguard
    missing = [k for k, v in final_settings.items() if v is None]
    if missing:
        raise ValueError(f"Missing required configuration values: {missing}. Provide them via JSON or CLI flags.")

    return final_settings
    
def get_runpod_s3_client(settings):
    # Pull keys dynamically using the variable names defined in your config
    access_key = os.getenv(settings["s3_access_var"])
    secret_key = os.getenv(settings["s3_secret_var"])
    datacenter_id = settings["datacenter_id"]
    
    if not access_key or not secret_key:
        raise ValueError(f"Missing environment variables: {settings['s3_access_var']} or {settings['s3_secret_var']}")

    endpoint_url = f"https://s3api-{datacenter_id.lower()}.runpod.io/"

    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=datacenter_id.lower(),
        config=Config(signature_version='s3v4')
    )

def upload_script(settings):
    local_path = settings["local_file_path"]
    volume_id = settings["network_volume_id"]
    remote_key = settings["remote_s3_key"]

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Could not find local script: {local_path}")
        
    s3_client = get_runpod_s3_client(settings)
    
    print(f"Uploading {local_path} directly to RunPod volume '{volume_id}' ({settings['datacenter_id']})...")
    try:
        s3_client.upload_file(
            Filename=local_path,
            Bucket=volume_id, 
            Key=remote_key
        )
        print("✅ Upload complete! The script is safely stored on your permanent network volume.")
        print(f"It will be accessible inside your pods at: /workspace/{remote_key}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")

def main():
    settings = parse_arguments()
    upload_script(settings)

if __name__ == "__main__":
    main()