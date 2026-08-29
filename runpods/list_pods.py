"""
List all currently running/active RunPod pods.
Usage: python list_pods.py
"""
import runpod
from read_config import getConfig


def main():
    cfg = getConfig()
    api_key = cfg["runpod_auth"]["api_key_env_var"]
    runpod.api_key = api_key

    pods = runpod.get_pods()

    if not pods:
        print("No pods found.")
        return

    print(f"{'ID':<22} {'Name':<25} {'Status':<12} {'GPU':<30} {'Image'}")
    print("-" * 110)
    for pod in pods:
        pod_id   = pod.get("id", "")
        name     = pod.get("name", "")
        status   = pod.get("desiredStatus") or pod.get("status", "")
        gpu      = pod.get("machine", {}).get("gpuDisplayName", "") or pod.get("gpuDisplayName", "")
        image    = pod.get("imageName", "")
        print(f"{pod_id:<22} {name:<25} {status:<12} {gpu:<30} {image}")


if __name__ == "__main__":
    main()
