import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

def get_dataloaders(batch_size=128, num_workers=2, data_dir='./data'):
    """
    Downloads CIFAR-10, applies basic transforms, splits the training set into
    base_train and calibration sets, and returns DataLoaders for all three sets.
    
    Key design decision: base_train uses augmentation (random crop, flip) for training,
    but calibration uses deterministic test transforms for consistent feature extraction.
    
    Returns:
        base_train_loader
        calibration_loader
        test_loader
    """
    
    # Training transforms with data augmentation (only for base_train)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Test/evaluation transforms — no augmentation (for calibration and test)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load the full training set twice: once with each transform
    train_dataset_aug = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    train_dataset_eval = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_test
    )
    
    # Load test dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # Split into 40k base_train + 10k calibration using a fixed seed
    base_train_size = 40000
    calibration_size = 10000

    # Use random_split just to get the indices, then apply correct transforms
    temp_base, temp_cal = random_split(
        train_dataset_aug,
        [base_train_size, calibration_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    # base_train: uses augmentation transforms (for training)
    base_train = Subset(train_dataset_aug, temp_base.indices)
    
    # calibration: uses deterministic test transforms (for consistent feature extraction)
    calibration = Subset(train_dataset_eval, temp_cal.indices)
    
    # Create DataLoaders
    base_train_loader = DataLoader(
        base_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    calibration_loader = DataLoader(
        calibration, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return base_train_loader, calibration_loader, test_loader

if __name__ == "__main__":
    base_loader, cal_loader, test_loader = get_dataloaders()
    
    print(f"Base-train images: {len(base_loader.dataset)}")
    print(f"Calibration images: {len(cal_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")
