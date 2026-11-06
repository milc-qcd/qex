#!/usr/bin/env python3
"""Install the pinned official LLVM SYCL compiler/runtime with its CUDA adapter."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION = "v7.2.0"
URL = f"https://github.com/intel/llvm/releases/download/{VERSION}/sycl_linux.tar.gz"
SHA256 = "385586a25e75ac727883f0be5e442dec0ee81b3c20f9a29037a79921ad2f1728"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    opt = Path(__file__).resolve().parents[2] / "opt"
    parser.add_argument("--prefix", type=Path, default=opt / f"llvm-sycl-{VERSION}-cuda")
    parser.add_argument("--cache", type=Path, default=opt / "downloads")
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    if prefix.exists():
        manifest = prefix / "qex-install-manifest.json"
        if not manifest.is_file() or json.loads(manifest.read_text()).get("sha256") != SHA256:
            parser.error(f"Refusing to overwrite an unrelated installation: {prefix}")
        print(f"Verified installation manifest: {prefix}")
    else:
        args.cache.mkdir(parents=True, exist_ok=True)
        archive = args.cache / f"llvm-sycl-{VERSION}-linux.tar.gz"
        if not archive.exists():
            part = archive.with_suffix(".download")
            with urllib.request.urlopen(URL, timeout=60) as source, part.open("wb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            part.replace(archive)
        if digest(archive) != SHA256:
            parser.error(f"Release checksum mismatch: {archive}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="qex-sycl-install-", dir=prefix.parent) as stage:
            staged = Path(stage) / "toolchain"
            staged.mkdir()
            # The archive is checksum-pinned to the official release. Also
            # reject paths or links that could escape the new installation.
            with tarfile.open(archive) as contents:
                for member in contents.getmembers():
                    destination = (staged / member.name).resolve()
                    if not destination.is_relative_to(staged):
                        parser.error(f"Archive path escapes prefix: {member.name}")
                    if member.issym() or member.islnk():
                        parent = destination.parent if member.issym() else staged
                        if not (parent / member.linkname).resolve().is_relative_to(staged):
                            parser.error(f"Archive link escapes prefix: {member.name}")
            subprocess.run(["tar", "--no-same-owner", "-xzf", str(archive), "-C", str(staged)], check=True)
            (staged / "qex-install-manifest.json").write_text(json.dumps({
                "installed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "release": f"https://github.com/intel/llvm/releases/tag/{VERSION}",
                "asset_url": URL, "sha256": SHA256,
            }, indent=2) + "\n")
            staged.rename(prefix)
    for name in ["bin/clang++", "bin/sycl-ls", "lib/libsycl.so", "lib/libur_adapter_cuda.so"]:
        if not (prefix / name).exists():
            parser.error(f"Incomplete installation: {prefix / name}")
    subprocess.run([str(prefix / "bin/clang++"), "--version"], check=True)
    print(f"Installed compiler and CUDA-enabled SYCL runtime: {prefix}")
    print("NVIDIA driver libraries and a GPU allocation are required for execution.")


if __name__ == "__main__":
    main()
