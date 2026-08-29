"""
Runs on the RunPod container.
Reads all config from environment variables set by deploy_and_execute.py.

Steps:
  1. Download project zip from S3 via presigned GET URL
  2. Unzip into /workspace
  3. Run python <dirname>/main.py, capturing stdout + stderr
  4. Zip stdout.txt + stderr.txt into results.zip
  5. Upload results.zip to S3 via presigned PUT URL
"""
import os
import sys
import zipfile
import subprocess
import urllib.request
import tempfile


def download(url, dest):
    print(f"[wrapper] Downloading to {dest} ...")
    urllib.request.urlretrieve(url, dest)


def upload_zip(zip_path, put_url):
    print(f"[wrapper] Uploading {zip_path} to S3 ...")
    with open(zip_path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(put_url, data=data, method="PUT")
    req.add_header("Content-Type", "application/zip")
    with urllib.request.urlopen(req) as resp:
        print(f"[wrapper] Upload complete (HTTP {resp.status})")

def install_dependencies(workdir, dir_name):
    requirements_path = os.path.join(workdir, dir_name, "requirements.txt")
    if not os.path.exists(requirements_path):
        print(f"[wrapper] No requirements.txt found in {dir_name}, skipping dependencies installation.")
        return
    print(f"[wrapper] Installing dependencies ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], cwd=os.path.join(workdir, dir_name))

def main():
    zip_url    = os.environ["ZIP_URL"]
    result_url = os.environ["RESULT_PUT_URL"]
    dir_name   = os.environ["DIR_NAME"]
    model_get_url = os.environ["MODEL_GET_URL"]
    model_put_best_url = os.environ["MODEL_PUT_BEST_URL"]
    model_put_last_url = os.environ["MODEL_PUT_LAST_URL"]
    run_id     = os.environ.get("RUN_ID", "unknown")
    
    workdir = "/workspace"
    os.makedirs(workdir, exist_ok=True)

    if model_get_url and model_get_url != "None":
        print(f"[wrapper] Downloading model from {model_get_url} ...")
        download(model_get_url, os.path.join(workdir, "model.pth"))

    # 1. Download + unzip project
    zip_path = os.path.join(workdir, "project.zip")
    download(zip_url, zip_path)
    print("[wrapper] Unzipping ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(workdir)

    # 2. Install dependencies
    install_dependencies(workdir, dir_name)

    # 2. Run main.py — stream output line-by-line while accumulating for the zip
    import threading

    main_script = os.path.join(workdir, dir_name, "main.py")
    print(f"[wrapper] Running {main_script} ...")

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def stream(pipe, lines: list[str], dest):
        for raw in iter(pipe.readline, ""):
            dest.write(raw)
            dest.flush()
            lines.append(raw)
        pipe.close()

    proc = subprocess.Popen(
        [sys.executable, "-u", main_script],
        cwd=os.path.join(workdir, dir_name),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    t_out = threading.Thread(target=stream, args=(proc.stdout, stdout_lines, sys.stdout))
    t_err = threading.Thread(target=stream, args=(proc.stderr, stderr_lines, sys.stderr))
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()
    returncode = proc.wait()

    print(f"[wrapper] exit code: {returncode}")

    stdout_text = "".join(stdout_lines)
    stderr_text = "".join(stderr_lines)

    # 3. Zip stdout + stderr into results.zip
    results_zip_path = os.path.join(workdir, "results.zip")
    with zipfile.ZipFile(results_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("stdout.txt",    stdout_text)
        zf.writestr("stderr.txt",    stderr_text)
        zf.writestr("exit_code.txt", str(returncode))
        zf.writestr("run_id.txt",    run_id)

    # 4. Upload results.zip to S3
    upload_zip(results_zip_path, result_url)
    print("[wrapper] Done.")


if __name__ == "__main__":
    main()
