from dataclasses import dataclass


@dataclass
class DatasetConfig:
    num_points: int = 2048
    num_classes: int = 40


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class ModelConfig:
    name: str = "DummyModel"


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    save_every: int = 1


class Config:

    dataset = DatasetConfig()
    training = TrainingConfig()
    model = ModelConfig()
    checkpoint = CheckpointConfig()


cfg = Config()