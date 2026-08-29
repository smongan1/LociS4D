"""
Zip a local project directory, upload it to AWS S3, launch a RunPod pod
that runs deployment_wrapper.py to execute the project, then downloads
results.zip (stdout.txt, stderr.txt, exit_code.txt).

The pod also receives MODEL_PUT_URL so main.py can push a trained model
directly to S3 without needing AWS credentials on the pod.

Usage:
    python deploy_and_execute.py --dir <path/to/project> [--cpu] [--zip-key <key>] [--model-key <key>]

The directory must contain a main.py at its root.
All config comes from configs/config.json via read_config.py.
"""
import os
import sys
import time
import base64
import zipfile
import argparse
import tempfile
from datetime import datetime, timezone
import runpod

from read_config import getConfig
from presigned_url import create_presigned_url, get_aws_s3_client

URL_TIMEOUT = 9000
WRAPPER_LOCAL  = os.path.join(os.path.dirname(__file__), "deployment_wrapper.py")
WRAPPER_S3_KEY = "runner/deployment_wrapper.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zip a directory, upload to S3, and execute on a RunPod pod."
    )
    parser.add_argument("--dir",       required=True,
                        help="Local directory to zip and deploy")
    parser.add_argument("--cpu",       action="store_true",
                        help="Use a CPU-only pod (cheaper, no GPU)")
    parser.add_argument("--zip-key",   help="S3 key for the zip (default: deployments/<dirname>.zip)")
    parser.add_argument("--model-key", help="S3 key for the model (default: models/<dirname>/best_model.pth)")
    parser.add_argument("--model-key-last", help="S3 key for the last model (default: models/<dirname>/last_model.pth)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous run (default: False)")
    parser.add_argument("--resume_best",action="store_true",
                        help="Best model to resume (default: False)")
    return parser.parse_args()


def zip_directory(dir_path):
    """Zip a directory into a temp file. Returns (tmp_path, dir_name)."""
    dir_path = os.path.abspath(dir_path)
    dir_name = os.path.basename(dir_path)
    tmp      = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, prefix=f"{dir_name}_")

    print(f"Zipping '{dir_path}' ...")
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "venv", ".venv", "node_modules")]
            for file in files:
                if file.endswith(".pyc"):
                    continue
                abs_path = os.path.join(root, file)
                arc_path = os.path.join(dir_name, os.path.relpath(abs_path, dir_path))
                zf.write(abs_path, arc_path)

    size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    print(f"Zip created: {size_mb:.2f} MB")
    return tmp.name, dir_name


def upload_file(local_path, s3_key, cfg):
    s3_client = get_aws_s3_client(cfg)
    bucket    = cfg["infrastructure_s3"]["bucket_name"]
    print(f"Uploading to s3://{bucket}/{s3_key} ...")
    s3_client.upload_file(Filename=local_path, Bucket=bucket, Key=s3_key)
    print("Upload complete.")


def presigned_put(key, cfg, expiration=URL_TIMEOUT, content_type="application/octet-stream"):
    s3_client = get_aws_s3_client(cfg)
    bucket    = cfg["infrastructure_s3"]["bucket_name"]
    return s3_client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=expiration,
    )



def get_pod_uptime(pod_id):
    """
    Return the pod's current uptimeInSeconds, or None if unreachable.
    The SDK's built-in get_pod query doesn't request uptimeInSeconds in its
    runtime block, so we fire our own minimal query that does.
    """
    try:
        from runpod.api.graphql import run_graphql_query
        query = f"""
        query {{
            pod(input: {{ podId: "{pod_id}" }}) {{
                runtime {{
                    uptimeInSeconds
                }}
            }}
        }}
        """
        resp    = run_graphql_query(query)
        pod     = (resp.get("data") or {}).get("pod") or {}
        runtime = pod.get("runtime") or {}
        return runtime.get("uptimeInSeconds")
    except Exception:
        return None


def poll_and_download(s3_key, local_path, cfg, pod_id, timeout=6000, poll_interval=30):
    """
    Wait until results.zip appears in S3, then download it.

    If pod_id is supplied, the pod's uptime is checked every cycle.
    Whenever the pod reports a *higher* uptime than the previous check
    (i.e. it is still alive and running), elapsed is reset to 0 so the
    timeout only fires on true silence — not just slow training runs.

    Note: RunPod does not expose a container-log API (github.com/runpod/
    runpod-python/issues/400), so uptime is the only liveness signal.
    """
    s3_client    = get_aws_s3_client(cfg)
    bucket       = cfg["infrastructure_s3"]["bucket_name"]
    last_uptime  = None
    elapsed      = 0

    print(f"Polling s3://{bucket}/{s3_key} ...")

    while elapsed < timeout:
        # ── S3 result check ───────────────────────────────────────────────
        try:
            s3_client.download_file(bucket, s3_key, local_path)
            print(f"Downloaded to: {local_path}")
            return
        except Exception as e:
            if "NoSuchKey" not in str(e) and "404" not in str(e):
                raise

        uptime = get_pod_uptime(pod_id)
        if uptime is not None and (last_uptime is None or uptime > last_uptime):
            if last_uptime is not None:
                # Pod is still alive and its clock is ticking — reset
                print(f"  Pod alive (uptime {uptime}s) — resetting timeout.")
                elapsed = 0
            last_uptime = uptime
        elif uptime is None and last_uptime is not None:
            print("  Warning: pod no longer reachable — may have crashed.")

        time.sleep(poll_interval)
        elapsed += poll_interval
        uptime_str = f"  pod uptime {last_uptime}s" if last_uptime is not None else ""
        print(f"  Waiting for results... ({elapsed}s idle){uptime_str}")

    raise TimeoutError(
        f"No results in S3 after {timeout}s of pod inactivity — "
        "check pod logs on the RunPod dashboard."
    )


def print_results(results_zip_path):
    with zipfile.ZipFile(results_zip_path, "r") as zf:
        stdout   = zf.read("stdout.txt").decode(errors="replace")
        stderr   = zf.read("stderr.txt").decode(errors="replace")
        exitcode = zf.read("exit_code.txt").decode().strip()

    print("\n========== STDOUT ==========")
    print(stdout or "(empty)")
    if stderr:
        print("\n========== STDERR ==========")
        print(stderr)
    print(f"\n========== EXIT CODE: {exitcode} ==========\n")


def main():
    args   = parse_args()
    cfg    = getConfig()
    infra  = cfg["infrastructure_runpods"]
    rp_cfg = cfg["runpod_auth"]

    runpod.api_key = rp_cfg["api_key_env_var"]

    dir_path = os.path.abspath(args.dir)
    if not os.path.isdir(dir_path):
        print(f"Error: '{dir_path}' is not a directory.")
        sys.exit(1)
    if not os.path.exists(os.path.join(dir_path, "main.py")):
        print(f"Error: No main.py found in '{dir_path}'.")
        sys.exit(1)

    dir_name  = os.path.basename(dir_path)
    run_id    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_key   = args.zip_key   or f"deployments/{dir_name}.zip"
    model_key_best = args.model_key or f"models/{dir_name}/best_model.pth"
    model_key_last = args.model_key_last or f"models/{dir_name}/last_model.pth"
    result_key = f"results/{dir_name}/{run_id}_results.zip"
    local_results_path = os.path.join(os.path.dirname(dir_path), f"{dir_name}_{run_id}_results.zip")
    print(f"Run ID: {run_id}")

    # 1. Zip + upload project
    zip_path, dir_name = zip_directory(dir_path)
    try:
        upload_file(zip_path, zip_key, cfg)
    finally:
        os.unlink(zip_path)

    # 2. Upload deployment_wrapper.py
    upload_file(WRAPPER_LOCAL, WRAPPER_S3_KEY, cfg)

    # 3. Generate presigned URLs
    bucket         = cfg["infrastructure_s3"]["bucket_name"]
    zip_url        = create_presigned_url(WRAPPER_S3_KEY, expiration=URL_TIMEOUT, cfg=cfg)  # wrapper GET
    wrapper_url    = create_presigned_url(WRAPPER_S3_KEY, expiration=URL_TIMEOUT, cfg=cfg)
    zip_get_url    = create_presigned_url(zip_key,        expiration=URL_TIMEOUT, cfg=cfg)
    result_put_url = presigned_put(result_key, cfg, expiration=URL_TIMEOUT, content_type="application/zip")
    model_put_best_url  = presigned_put(model_key_best,  cfg, expiration=URL_TIMEOUT, content_type="application/octet-stream")
    model_put_last_url  = presigned_put(model_key_last,  cfg, expiration=URL_TIMEOUT, content_type="application/octet-stream")
    model_get_url = None
    if args.resume:
        model_get_url = create_presigned_url(model_key_last, expiration=URL_TIMEOUT, cfg=cfg)
    elif args.resume_best:
        model_get_url = create_presigned_url(model_key_best, expiration=URL_TIMEOUT, cfg=cfg)

    print("Presigned URLs generated.")
    print(f"  Results : s3://{bucket}/{result_key}")
    print(f"  Model   : s3://{bucket}/{model_key_best}")
    print(f"  Model   : s3://{bucket}/{model_key_last}")
    if model_get_url:
        print(f"  Resume Model   : {model_get_url}")

    # 4. Pod startup: download wrapper, run it (all logic is inside the wrapper)
    startup_script = (
        "#!/bin/bash\n"
        "set -e\n"
        f'python -c "import urllib.request; urllib.request.urlretrieve(\'$WRAPPER_URL\', \'/tmp/deployment_wrapper.py\')"\n'
        "python /tmp/deployment_wrapper.py\n"
    )
    b64_script  = base64.b64encode(startup_script.encode()).decode()
    startup_cmd = f"bash -c 'echo {b64_script} | base64 -d | bash'"

    env_vars = {
        "WRAPPER_URL":    wrapper_url,
        "ZIP_URL":        zip_get_url,
        "RESULT_PUT_URL": result_put_url,
        "MODEL_PUT_BEST_URL":  model_put_best_url,
        "MODEL_PUT_LAST_URL":  model_put_last_url,
        "MODEL_GET_URL":  model_get_url,
        "DIR_NAME":       dir_name,
        "RUN_ID":         run_id,
    }

    # 5. Launch pod
    if args.cpu:
        instance_id = infra.get("cpu_instance_id", "cpu3c-2-4")
        print(f"Launching CPU pod (instance={instance_id})...")
        pod = runpod.create_pod(
            name=f"deploy-{dir_name}",
            image_name=infra["container_image"],
            instance_id=instance_id,
            docker_args=startup_cmd,
            env=env_vars,
        )
    else:
        print(f"Launching GPU pod ({infra['gpu_type_id']})...")
        pod = runpod.create_pod(
            name=f"deploy-{dir_name}",
            image_name=infra["container_image"],
            gpu_type_id=infra["gpu_type_id"],
            cloud_type=infra.get("cloud_type", "SECURE"),
            gpu_count=infra.get("gpu_count", 1),
            docker_args=startup_cmd,
            env=env_vars,
        )

    pod_id = pod["id"]
    print(f"Pod launched. ID: {pod_id}")

    try:
        # 6. Poll S3 until results.zip appears, download + print
        poll_and_download(result_key, local_results_path, cfg, pod_id)
        print_results(local_results_path)

    except Exception as e:
        print(f"Error: {e}")
        raise

    finally:
        print(f"Terminating pod {pod_id}...")
        runpod.terminate_pod(pod_id)
        print("Pod terminated.")


if __name__ == "__main__":
    main()
