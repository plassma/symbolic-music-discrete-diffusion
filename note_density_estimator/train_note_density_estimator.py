import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from load_midi import load_data, to_features
from sklearn.model_selection import train_test_split
from key_estimator import SimpleNoteDensityEstimator

input_size = 90
rnn_hidden_size = 128
hidden_size = 512
pitch_idxs = np.arange(rnn_hidden_size)
dur_idxs = np.arange(rnn_hidden_size, input_size)


def train(model, device, train_loader, test_dataloader, optimizer, epochs):
    loss_func = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    for i in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device).float(), target.to(device)
            data = data.view(-1, 16, data.shape[-1])

            #count onsets for note-density
            target = (data.argmax(-1) > 1).sum(-1)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            epoch_loss += loss.sum().item()
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(train_loader.dataset) * train_loader.batch_size
        print(f"epoch{i}, loss: {epoch_loss}")
        test_losses += [test(model, "cuda", test_dataloader)]
        train_losses += [epoch_loss]

    return train_losses, test_losses


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0.0
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device)
            data = data.view(-1, 16, data.shape[-1])
            target = (data.argmax(-1) > 1).sum(-1)
            output = model(data)
            test_loss += loss_func(output, target).sum().item()
            correct += (output.argmax(-1) == target).sum() / data.shape[0]


    test_loss = test_loss / len(test_loader.dataset) * test_loader.batch_size

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


class KeyDataset(Dataset):
    def __init__(self, hidden_states, labels):
        self.hidden_states = hidden_states
        self.distinct_labels = ['A', 'A#', 'A#m', 'Am', 'B', 'Bm', 'C', 'C#', 'C#m', 'Cm', 'D', 'D#', 'D#m', 'Dm', 'E', 'E#', 'E#m', 'Em', 'F#', 'F#m', 'G', 'G#', 'G#m', 'Gm']#sorted(list(set(labels))) # sort to have stable indices!
        self.labels = [self.distinct_labels.index(l) for l in labels]
    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, i):
        return self.hidden_states[i], self.labels[i]


if __name__ == "__main__":
    #todo: uncouple this from
    data, labels = load_data(par_dir="/home/plassma/Desktop/JKU/Master/music_informatics_team_mozart")
    data = to_features(data)
    num_classes = len(set(labels))

    REC_IN_OUT_SIZE = 128

    #use KeyDataset, ignore labels, calc note-density from data
    dataset = KeyDataset(data, labels)
    rnn = SimpleNoteDensityEstimator(17).cuda()


    print("use this to decode with model: ", dataset.distinct_labels)

    train_set, test_set = train_test_split(dataset)

    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=1)

    optimizer = SGD(rnn.parameters(), 0.001)
    train(rnn, "cuda", train_dataloader, test_dataloader, optimizer, 30)
    test(rnn, 'cuda', test_dataloader)
    torch.save(rnn.state_dict(), "../note_density_estimator.pth")
