from __future__ import annotations

import argparse
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


SOURCE_PAGE_URL = "https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/"
ARCHIVE_URL = "https://go.criteo.net/criteo-research-attribution-dataset.zip"
LICENSE_NAME = "CC BY-NC-SA 4.0"
EXPECTED_DATASET_FILE = "criteo_attribution_dataset.tsv.gz"


def format_bytes(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n_bytes} B"


def download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as f:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        chunk_size = 1024 * 1024

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            if total > 0:
                pct = 100.0 * downloaded / total
                print(
                    f"\rDownloading {dest.name}: {format_bytes(downloaded)} / {format_bytes(total)} ({pct:5.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\rDownloading {dest.name}: {format_bytes(downloaded)}", end="", flush=True)

    print()


def extract_archive(archive_path: Path, out_dir: Path, *, force: bool) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        members = zf.namelist()
        if not members:
            raise RuntimeError(f"Archive {archive_path} is empty.")

        for member in members:
            target = out_dir / member
            if target.exists() and not force:
                continue
            zf.extract(member, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/raw/criteo_attrib")
    ap.add_argument("--archive_name", type=str, default="criteo-research-attribution-dataset.zip")
    ap.add_argument("--force", action="store_true", help="Redownload and re-extract even if files already exist.")
    ap.add_argument("--keep_archive", action="store_true", help="Keep the downloaded zip after extraction.")
    ap.add_argument(
        "--accept_criteo_nc_sa_license",
        action="store_true",
        help=f"Required. Confirms you accept the dataset terms published by Criteo under {LICENSE_NAME}.",
    )
    args = ap.parse_args()

    if not args.accept_criteo_nc_sa_license:
        raise SystemExit(
            "Refusing to download without explicit license acceptance.\n"
            f"Review the dataset page: {SOURCE_PAGE_URL}\n"
            f"License: {LICENSE_NAME}\n"
            "Re-run with: --accept_criteo_nc_sa_license"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / args.archive_name
    dataset_path = out_dir / EXPECTED_DATASET_FILE

    if dataset_path.exists() and not args.force:
        print(f"Dataset already present: {dataset_path}")
        print("Use --force to redownload and re-extract.")
        return

    if archive_path.exists() and args.force:
        archive_path.unlink()

    print("Source page:", SOURCE_PAGE_URL)
    print("Archive URL:", ARCHIVE_URL)
    print("Output dir:", out_dir)

    try:
        if not archive_path.exists():
            download_file(ARCHIVE_URL, archive_path)
        else:
            print(f"Using existing archive: {archive_path}")
    except urllib.error.URLError as exc:
        raise SystemExit(
            "Automatic download failed.\n"
            f"Try opening the official page in a browser and downloading manually:\n{SOURCE_PAGE_URL}\n"
            f"Underlying error: {exc}"
        ) from exc

    print(f"Extracting {archive_path.name} into {out_dir} ...")
    extract_archive(archive_path, out_dir, force=bool(args.force))

    if not dataset_path.exists():
        raise SystemExit(
            f"Extraction completed, but {dataset_path} was not found.\n"
            "Inspect the extracted files manually."
        )

    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)

    print("Done.")
    print("Dataset:", dataset_path)
    print("Next step:")
    print(
        "python -m src.env.make_criteo_attrib_memmap_full "
        f"--inp {dataset_path} "
        "--out_dir data/processed/criteo_full_k50_d64_real_split80 "
        "--k_cap 50 --d_hash 64 --delta_seconds 3600 "
        "--censor_seconds $((5000*3600)) --d_max 5000 --train_frac 0.8 --split_seed 123"
    )


if __name__ == "__main__":
    main()
