from pathlib import Path


def find_modelnet40():

    candidates = [

        # Local
        Path("datasets/ModelNet40"),

        # Colab
        Path("/content/PointCloudAnalysis/datasets/ModelNet40"),

        # Kaggle
        Path("/kaggle/input/datasets/balraj98/modelnet40-princeton-3d-object-dataset/ModelNet40"),

        Path("/kaggle/input/modelnet40-princeton-3d-object-dataset/ModelNet40"),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    kaggle_root = Path("/kaggle/input")

    if kaggle_root.exists():

        for folder in kaggle_root.rglob("*"):

            if not folder.is_dir():
                continue

            try:

                classes = [
                    x for x in folder.iterdir()
                    if x.is_dir()
                ]

                if len(classes) >= 40:
                    return str(folder)

            except Exception:
                pass

    raise FileNotFoundError(
        "Unable to locate ModelNet40 dataset."
    )


DATASET_ROOT = find_modelnet40()