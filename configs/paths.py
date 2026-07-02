from pathlib import Path


def find_modelnet40():

    candidates = [

        Path("datasets/ModelNet40"),

        Path("/content/PointCloudAnalysis/datasets/ModelNet40"),

        Path("/kaggle/input/datasets/balraj98/modelnet40-princeton-3d-object-dataset/ModelNet40"),

        Path("/kaggle/input/modelnet40-princeton-3d-object-dataset/ModelNet40"),
    ]

    for path in candidates:

        if path.exists():
            return str(path)

    kaggle = Path("/kaggle/input")

    if kaggle.exists():

        for folder in kaggle.rglob("*"):

            if not folder.is_dir():
                continue

            try:

                children = [
                    x for x in folder.iterdir()
                    if x.is_dir()
                ]

                if len(children) >= 40:
                    return str(folder)

            except Exception:
                pass

    raise FileNotFoundError(
        "Unable to locate ModelNet40 dataset."
    )


DATASET_ROOT = find_modelnet40()