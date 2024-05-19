import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data import MNISTDataLoader
from model import LeNet5
import os
from tqdm import tqdm
from datetime import datetime
torch.backends.cudnn.enabled = False # Necesito solucionar esto

class Train():
    def __init__(self, batch_size=4, epochs=2, lr=0.001, log_dir_base="./runs", model_dir="./save_models", root='./data'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LeNet5().to(self.device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.log_dir_base = log_dir_base
        self.model_dir = model_dir
        self.root = root
        os.makedirs(self.log_dir_base, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.current_time = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir_base, "LeNet5_MNIST", self.current_time)
        self.data_loader = MNISTDataLoader(batch_size=self.batch_size, root=self.root)
        self.train_loader = self.data_loader.get_loader(train=True)
        self.test_loader = self.data_loader.get_loader(train=False)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.best_accuracy = 0.0
        self.best_model_path = os.path.join(self.model_dir, 'best_model')
                
    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        # Calcular pérdida promedio durante la época
        epoch_loss = running_loss / len(self.train_loader)
        self.writer.add_scalar('training loss', epoch_loss, epoch)

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        self.writer.add_scalar('Loss/test', test_loss, epoch)
        self.writer.add_scalar('Accuracy/test', accuracy, epoch)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.0f}%)\n')
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.model.state_dict(), self.best_model_path + "_" + self.current_time + ".pth")

    def run(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.test(epoch)
        self.writer.close()