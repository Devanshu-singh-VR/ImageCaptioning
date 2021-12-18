import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data import get_loader
from model import CNNtoRNN


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="images",
        annotation_file="Book1.csv",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, dataset, num_layers, device).to(device)
    loss_f = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        print(f"Epochs[{epoch}/{num_epochs}]")
        losses = []
        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            outputs = model(imgs, captions)

            optimizer.zero_grad()

            loss = loss_f(
                outputs[1:].reshape(-1, outputs.shape[2]), captions[1:].reshape(-1)
            )
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            # test the caption model
            model.eval()
            print(model.image_to_caption(imgs))
            model.train()


        print(f'loss for epoch {epoch} is {sum(losses) / len(losses)}')


if __name__ == "__main__":
    train()