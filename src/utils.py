from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, List


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    image_size: int = 224,
    model_type: str = "resnet",  # or "custom"
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Loads training and testing data from ImageFolder and returns DataLoaders and class names.
    Applies model-specific transforms:
    - For 'resnet': use ImageNet normalization and stronger augmentations.
    - For 'custom': use basic grayscale normalization.
    """

    if model_type == "resnet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),  # small, safe
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    elif model_type == "custom":
        mean = [0.5]
        std = [0.5]

        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                # Uncomment if you want minimal rotation for CNN
                # transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'resnet' or 'custom'.")

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
