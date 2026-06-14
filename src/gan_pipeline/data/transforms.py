from torchvision import transforms


def get_transforms(image_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
