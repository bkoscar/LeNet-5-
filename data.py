import torch
from torchvision import datasets, transforms
import cv2
import numpy as np

class MNISTDataLoader:
    def __init__(self, batch_size=64, root='./data'):
        self.batch_size = batch_size
        self.root = root
        # Transformaciones para las imágenes
        self.transform = transforms.Compose([
             transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # Descargar los datos
        self.train_dataset = datasets.MNIST(root=self.root, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root=self.root, train=False, download=True, transform=self.transform)
        # Crear los dataloaders
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    def get_loader(self, train=True):
        return self.train_loader if train else self.test_loader 

 
    def show_loader(self, train = True):
        loader = self.train_loader if train else self.test_loader
        data_iter = iter(loader)
        data, target = next(data_iter)
        print(f"Data shape: {data.shape}")
        print(f"Target shape: {target.shape}")
                
    def show_images(self, train = True, num_images=5):
        loader = self.train_loader if train else self.test_loader
        data_iter = iter(loader)
        images, labels = next(data_iter)
        images = np.array(images)
        labels = np.array(labels)
        for i in range(num_images):
            image = images[i][0]
            label = labels[i]
            image = cv2.resize(image, (500, 500))
            cv2.imshow(f"Label: {label}", image)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        


# Uso de la clase MNISTDataLoader
if __name__ == "__main__":
    data_loader = MNISTDataLoader(batch_size = 5, root = './data')
    data_loader.show_loader(train=True)
    data_loader.show_images(train=True)
    # data_loader.show_images(train=False)