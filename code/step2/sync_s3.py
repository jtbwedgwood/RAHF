import subprocess
import os


S3_BUCKET_URI = os.environ.get("S3_BUCKET_URI", "")

def _s3_sync(local_dir: str, remote_uri: str, timeout: int = 900):
    if not remote_uri:
        return
    subprocess.run(
        ["timeout", str(timeout), "aws", "s3", "sync", local_dir, remote_uri, "--only-show-errors"],
        check=True
    )

_s3_sync("/workspace/SCIT", S3_BUCKET_URI.rstrip('/'))