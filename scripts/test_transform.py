from data.modelnet_dataset import ModelNet40Dataset
from data.transforms import *

transform = Compose([
    Normalize(),
    RandomRotate(),
    RandomJitter(),
    ToTensor()
])

dataset = ModelNet40Dataset(
    root_dir="datasets/ModelNet40",
    split="train",
    transform=transform
)

points, label = dataset[0]

print(type(points))
print(points.shape)
print(points.dtype)

print(points.max())
print(points.min())