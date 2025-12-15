from typing import Tuple
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net


def create_dataloaders(batch_size: int = 4) -> Tuple[DataLoader, DataLoader, Tuple[str]]:
    """
    Create CIFAR-10 train/test dataloaders.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def train(model: Net, trainloader: DataLoader, epochs: int = 2) -> None:
    """
    Train the CNN model.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")


def evaluate(model: Net, testloader: DataLoader, classes: Tuple[str]) -> None:
    """
    Evaluate model accuracy with CIFAR-10 test set.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f} %")


def main() -> None:
    trainloader, testloader, classes = create_dataloaders()
    model = Net()
    train(model, trainloader)
    evaluate(model, testloader, classes)


if __name__ == "__main__":
    main()
