import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


def load_CIFAR10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 패딩 4 주고 랜덤 크롭
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Normalize 시키기 위해 tensor로 변환
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # per-pixel mean subtract 적용
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    train_set = datasets.CIFAR10(root="/data", train=True, transform=transform_train, download=True)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)

    test_set = datasets.CIFAR10(root="/data", train=False, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)

    return train_loader, test_loader

