from dataclasses import dataclass


@dataclass
class DatasetConfig:

    num_points: int = 2048
    num_classes: int = 40


@dataclass
class TrainingConfig:

    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-3
    num_workers: int = 2
    shuffle: bool = True


@dataclass
class CheckpointConfig:

    directory: str = "checkpoints"


@dataclass
class ModelConfig:

    name: str = "DummyModel"


class Config:

    dataset = DatasetConfig()
    training = TrainingConfig()
    checkpoint = CheckpointConfig()
    model = ModelConfig()


cfg = Config()