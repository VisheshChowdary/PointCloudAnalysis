from training.dataloader import create_dataloader

loader = create_dataloader(
    root_dir="datasets/ModelNet40",
    split="train",
    batch_size=8
)

print("=" * 50)

for points, labels in loader:

    print("Points :", points.shape)
    print("Labels :", labels.shape)

    print(points.dtype)

    print(labels)

    break