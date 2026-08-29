"""
Smoke test script executed on the RunPod container.
Prints a confirmation message and writes a result .txt back to S3
via a presigned PUT URL (no AWS credentials needed on the pod).

Usage (called by pod_runner.py):
    python deploy_smoke_test.py <presigned_put_url>
"""
import sys
import socket
import datetime
import urllib.request

# --- confirmation message ---
timestamp = datetime.datetime.utcnow().isoformat()
hostname  = socket.gethostname()
message   = f"Smoke test passed!\nhost={hostname}\nutc={timestamp}\n"
print(message)

# --- write result back to S3 via presigned PUT URL ---
if len(sys.argv) < 2:
    print("No presigned PUT URL provided — skipping S3 write-back.")
    sys.exit(0)

put_url = sys.argv[1]
data    = message.encode("utf-8")
req     = urllib.request.Request(put_url, data=data, method="PUT")
req.add_header("Content-Type", "text/plain")

try:
    with urllib.request.urlopen(req) as resp:
        print(f"Result written to S3 (HTTP {resp.status})")
except Exception as e:
    print(f"S3 write-back failed: {e}")
    sys.exit(1)
