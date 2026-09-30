from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "app" / "datasets" / "heart_disease"
URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / "processed.cleveland.data"
    if target.exists():
        print(f"Dataset already exists: {target}")
        return
    with urlopen(URL, timeout=30) as response:
        archive_bytes = response.read()
    archive_path = DATA_DIR / "heart_disease.zip"
    archive_path.write_bytes(archive_bytes)
    with ZipFile(archive_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("processed.cleveland.data"))
        target.write_bytes(archive.read(member))
    archive_path.unlink()
    print(f"Downloaded UCI Cleveland dataset: {target}")


if __name__ == "__main__":
    main()