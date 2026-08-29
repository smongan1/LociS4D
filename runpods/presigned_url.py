import os
import boto3
from botocore.config import Config
from read_config import getConfig


def get_aws_s3_client(cfg):
    s3_cfg = cfg["infrastructure_s3"]
    access_key = s3_cfg["s3_local_access_public_key"]
    secret_key = s3_cfg["s3_local_access_secret_key"]
    region = s3_cfg["region_name"]

    if not access_key or not secret_key:
        raise ValueError(
            f"Missing env vars: {s3_cfg['aws_access_key_env_var']} or {s3_cfg['aws_secret_key_env_var']}"
        )

    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(signature_version="s3v4"),
    )


def create_presigned_url(key, expiration=3600, cfg=None):
    """Generate a presigned GET URL for an object in the configured S3 bucket."""
    if cfg is None:
        cfg = getConfig()

    s3_client = get_aws_s3_client(cfg)
    bucket = cfg["infrastructure_s3"]["bucket_name"]

    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiration,
    )
    return url


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a presigned S3 URL.")
    parser.add_argument("--key", required=True, help="S3 object key")
    parser.add_argument("--expiration", type=int, default=3600, help="Seconds until URL expires")
    args = parser.parse_args()

    url = create_presigned_url(args.key, expiration=args.expiration)
    print(f"Presigned URL:\n{url}")
