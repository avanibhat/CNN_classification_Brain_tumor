from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, List


def get_dataloaders(
    data_dir: str, batch_size: int, image_size: int = 224
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Loads training and testing data from ImageFolder and returns DataLoaders and class names.
    Applies data augmentation on the training data to improve generalization.
    """
    # Training transforms with augmentation
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),  # randomly flip left-right
            # randomly rotate ±10 degrees transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # Testing transforms (NO augmentation)
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/Training", transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/Testing", transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes
    return train_loader, test_loader, class_names
