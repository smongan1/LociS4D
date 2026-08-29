import os
import argparse
import boto3
from botocore.config import Config
from read_config import getConfig
from presigned_url import get_aws_s3_client


def upload_script(local_path, s3_key, cfg=None):
    """Upload a local file to the configured AWS S3 bucket. Returns the S3 key."""
    if cfg is None:
        cfg = getConfig()

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Script not found: {local_path}")

    s3_client = get_aws_s3_client(cfg)
    bucket = cfg["infrastructure_s3"]["bucket_name"]

    print(f"Uploading '{local_path}' to s3://{bucket}/{s3_key} ...")
    s3_client.upload_file(Filename=local_path, Bucket=bucket, Key=s3_key)
    print(f"Upload complete: s3://{bucket}/{s3_key}")
    return s3_key


def main():
    parser = argparse.ArgumentParser(description="Upload a script to AWS S3.")
    parser.add_argument("--script", required=True, help="Local path to the script to upload")
    parser.add_argument("--key", help="S3 destination key (defaults to deployment_paths.remote_script_key from config)")
    args = parser.parse_args()

    cfg = getConfig()
    s3_key = args.key or cfg["deployment_paths"]["remote_script_key"]
    upload_script(args.script, s3_key, cfg=cfg)


if __name__ == "__main__":
    main()
