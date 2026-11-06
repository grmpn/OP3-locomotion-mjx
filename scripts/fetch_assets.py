"""Fetch only OP3 from a pinned Menagerie revision; verify every Git blob."""

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
REPOSITORY = "google-deepmind/mujoco_menagerie"
DESTINATION = Path(__file__).resolve().parents[1] / "assets" / "robotis_op3"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "op3-locomotion"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    tree = json.loads(
        fetch(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1")
    )
    if tree.get("truncated"):
        raise RuntimeError("Upstream tree was truncated")
    entries = [
        e for e in tree["tree"] if e["type"] == "blob" and e["path"].startswith("robotis_op3/")
    ]
    if not entries or not any(e["path"].endswith("LICENSE") for e in entries):
        raise RuntimeError("OP3 assets or license missing from upstream tree")

    def download(entry: dict) -> tuple[str, str]:
        relative = Path(entry["path"]).relative_to("robotis_op3")
        if ".." in relative.parts:
            raise ValueError("Invalid upstream path")
        path = DESTINATION / relative
        data = path.read_bytes() if path.exists() else b""

        def blob_sha(value: bytes) -> str:
            return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()

        if blob_sha(data) != entry["sha"]:
            data = fetch(
                f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{entry['path']}"
            )
            if blob_sha(data) != entry["sha"]:
                raise RuntimeError(f"Checksum mismatch: {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return str(relative), hashlib.sha256(data).hexdigest()

    with ThreadPoolExecutor(max_workers=6) as pool:
        hashes = dict(pool.map(download, entries))
    (DESTINATION / "provenance.json").write_text(
        json.dumps(
            {
                "repository": f"https://github.com/{REPOSITORY}",
                "revision": REVISION,
                "sha256": hashes,
                "modifications": [],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {len(hashes)} files at {REVISION} in {DESTINATION}")


if __name__ == "__main__":
    main()
