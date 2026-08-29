"""
List available GPU types on RunPod with current availability and pricing.
Usage: python list_gpus.py [--cloud SECURE|COMMUNITY] [--min-vram 24]
"""
import argparse
import runpod
from read_config import getConfig


def parse_args():
    parser = argparse.ArgumentParser(description="List available RunPod GPU resources.")
    parser.add_argument("--cloud",    default="SECURE", choices=["SECURE", "COMMUNITY", "ALL"],
                        help="Cloud type filter (default: SECURE)")
    parser.add_argument("--min-vram", type=int, default=0,
                        help="Minimum VRAM in GB to show (default: 0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = getConfig()

    api_key = cfg["runpod_auth"]["api_key_env_var"]
    runpod.api_key = api_key

    gpu_types = runpod.get_gpus()

    if not gpu_types:
        print("No GPU types returned.")
        return

    # Filter and sort
    results = []
    for gpu in gpu_types:
        vram = gpu.get("memoryInGb", 0) or 0
        if vram < args.min_vram:
            continue

        secure_avail    = gpu.get("secureCloud", False)
        community_avail = gpu.get("communityCloud", False)

        if args.cloud == "SECURE"    and not secure_avail:
            continue
        if args.cloud == "COMMUNITY" and not community_avail:
            continue

        results.append(gpu)

    results.sort(key=lambda g: g.get("memoryInGb", 0))

    if not results:
        print(f"No GPUs available matching filters (cloud={args.cloud}, min_vram={args.min_vram}GB).")
        return

    print(f"\nAvailable GPUs  [cloud={args.cloud}, min_vram={args.min_vram}GB]\n")
    print(f"{'GPU':<38} {'VRAM':>6}  {'Secure':>7}  {'Community':>10}  {'$/hr (secure)':>14}")
    print("-" * 85)
    for gpu in results:
        name      = gpu.get("displayName", gpu.get("id", ""))
        vram      = gpu.get("memoryInGb", "?")
        secure    = "yes" if gpu.get("secureCloud")    else "no"
        community = "yes" if gpu.get("communityCloud") else "no"
        price     = gpu.get("securePrice") or gpu.get("lowestPrice", {}).get("minimumBidPrice", "?")
        price_str = f"${price:.3f}" if isinstance(price, (int, float)) else str(price)
        print(f"{name:<38} {str(vram)+' GB':>6}  {secure:>7}  {community:>10}  {price_str:>14}")

    print()


if __name__ == "__main__":
    main()
