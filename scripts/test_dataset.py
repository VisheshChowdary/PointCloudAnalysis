from data.modelnet_dataset import ModelNet40Dataset

dataset = ModelNet40Dataset(
    root_dir="datasets/ModelNet40",
    split="train",
    num_points=2048
)

print("=" * 50)
print("Number of Classes :", len(dataset.classes))
print("Number of Samples :", len(dataset))
print("=" * 50)

points, label = dataset[0]

print("Point Shape :", points.shape)
print("Point Type  :", points.dtype)
print("Label       :", label)
print("Class Name  :", dataset.classes[label])

print("\nFirst 5 Points:")
print(points[:5])