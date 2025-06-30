from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_train_loader(dataset, data_root, batch_size, dataset_mean, dataset_std):
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ])
        data = datasets.MNIST(root=data_root+'/MNIST', train=True, transform=transform, download=True)
    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])
        data = datasets.ImageFolder(root=data_root+'/MNIST_M/train', transform=transform)
    else:
        raise Exception(f'There is no dataset named {str(dataset)}')
    
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=True)