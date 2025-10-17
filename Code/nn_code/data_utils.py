"""Data loading and preparation utilities."""
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from datasets.utils.logging import disable_progress_bar

# Disable progress bar for cleaner output
disable_progress_bar()

# Constants
BATCH_SIZE = 32


def load_datasets(partition_id: int, num_partitions: int):
    """Load and prepare the CIFAR-10 dataset for federated learning."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader


def visualize_dataset_sample(trainloader):
    """Visualize a batch of images from the dataset."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    batch = next(iter(trainloader))
    images, labels = batch["img"], batch["label"]
    images = images.permute(0, 2, 3, 1).numpy()
    images = images / 2 + 0.5

    fig, axs = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axs.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
            ax.axis("off")

    fig.tight_layout()
    plt.show()