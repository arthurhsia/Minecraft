#!/usr/bin/env python3
"""Download the MediaPipe Tasks model files into ``models/`` (standard library only).

    python scripts/download_models.py            # skip files that are already present
    python scripts/download_models.py --force    # re-download

Files:
  models/hand_landmarker.task   (~7.5 MB)
  models/face_landmarker.task   (~3.6 MB)

Run this once on the machine that runs the controller (needs internet once; the app itself
never uses the network).
"""
from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS: dict[str, str] = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/latest/face_landmarker.task",
}
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models"
MIN_BYTES = 100_000       # anything smaller is an error page, not a model


def _progress(name: str, done: int, total: int | None, t0: float) -> None:
    rate = done / max(1e-6, time.monotonic() - t0) / 1e6
    if total:
        pct = 100.0 * done / total
        bar = "#" * int(pct / 4)
        msg = f"\r  {name}: [{bar:<25}] {pct:5.1f}%  {done / 1e6:5.1f}/{total / 1e6:.1f} MB  {rate:4.1f} MB/s"
    else:
        msg = f"\r  {name}: {done / 1e6:5.1f} MB  {rate:4.1f} MB/s"
    sys.stdout.write(msg)
    sys.stdout.flush()


def download(url: str, dest: Path, timeout: float = 30.0) -> None:
    """Download ``url`` to ``dest`` atomically (``.part`` file, then rename)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "mcbio-model-downloader"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0) or None
        done = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            _progress(dest.name, done, total, t0)
    sys.stdout.write("\n")
    if done < MIN_BYTES or (total and done != total):
        tmp.unlink(missing_ok=True)
        raise IOError(f"incomplete download ({done} bytes" + (f" of {total})" if total else ")"))
    os.replace(tmp, dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download MediaPipe .task models into models/.")
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help=f"target folder (default {DEFAULT_DIR})")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    a = ap.parse_args(argv)
    target = Path(a.dir)
    target.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name, url in MODELS.items():
        dest = target / name
        if dest.exists() and dest.stat().st_size >= MIN_BYTES and not a.force:
            print(f"  {name}: already present ({dest.stat().st_size / 1e6:.1f} MB) - skipping")
            continue
        print(f"Downloading {name}\n  from {url}")
        try:
            download(url, dest)
            print(f"  saved {dest}")
        except urllib.error.HTTPError as e:
            failures += 1
            print(f"\n  ERROR: server answered HTTP {e.code} {e.reason} for {name}.\n"
                  "  The URL may have moved - check https://ai.google.dev/edge/mediapipe/solutions/"
                  "vision/hand_landmarker (or face_landmarker) and download it by hand into "
                  f"{target}", file=sys.stderr)
        except urllib.error.URLError as e:
            failures += 1
            reason = e.reason
            hint = ""
            if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE" in str(reason).upper():
                hint = ("\n  macOS python.org builds: run 'Install Certificates.command' from your "
                        "Python folder, or use the conda/Anaconda Python.")
            print(f"\n  ERROR: could not reach the server ({reason}). Are you online / behind a proxy?"
                  f"{hint}", file=sys.stderr)
        except (OSError, IOError) as e:
            failures += 1
            print(f"\n  ERROR writing {dest}: {e}", file=sys.stderr)
    if failures:
        print(f"\n{failures} model(s) failed to download.", file=sys.stderr)
        return 1
    print(f"\nAll models ready in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
