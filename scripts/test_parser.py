from pathlib import Path

from data.off_parser import OFFParser

parser = OFFParser(num_points=2048)

file = Path(
    "datasets/ModelNet40/airplane/train/airplane_0001.off"
)

points = parser.load(file)

print(points.shape)
print(points.dtype)
print(points[:5])