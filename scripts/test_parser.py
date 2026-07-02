from pathlib import Path

from data.off_parser import OFFParser


def main():

    print("=" * 60)
    print("TESTING OFF PARSER")
    print("=" * 60)

    parser = OFFParser(num_points=2048)

    file_path = Path(
        "datasets/ModelNet40/airplane/train/airplane_0001.off"
    )

    print(f"Loading: {file_path}")

    points = parser.load(file_path)

    print("\nSUCCESS")
    print("-" * 60)

    print("Shape :", points.shape)
    print("Type  :", points.dtype)

    print("\nFirst 5 Points")

    print(points[:5])

    print("\nParser Working Correctly")
    print("=" * 60)


if __name__ == "__main__":
    main()