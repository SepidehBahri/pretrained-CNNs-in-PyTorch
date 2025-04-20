from torchvision import transforms

def get_train_transforms(image_size=(224, 224), horizontal_flip=False, rotation_range=0):
    transform_list = [transforms.Resize(image_size)]

    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if rotation_range > 0:
        transform_list.append(transforms.RandomRotation(rotation_range))

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def get_test_transforms(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
