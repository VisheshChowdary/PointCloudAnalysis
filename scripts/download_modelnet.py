from pathlib import Path
import urllib.request
import zipfile
import shutil

URL = "https://modelnet.cs.princeton.edu/ModelNet40.zip"

root = Path("datasets")
root.mkdir(exist_ok=True)

zip_path = root / "ModelNet40.zip"

print("Downloading ModelNet40...")

urllib.request.urlretrieve(URL, zip_path)

print("Extracting...")

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(root)

zip_path.unlink()

print("Done!")
print("Dataset saved to:", root / "ModelNet40")