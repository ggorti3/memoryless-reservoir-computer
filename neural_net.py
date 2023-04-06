import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

class FeedForwardNN(nn.Module):
    def __init__(self, dim_system=3, dim_reservoir=300):
        super().__init__()
        self.dim_reservoir = dim_reservoir
        self.dim_system = dim_system
        self.W_in = nn.Linear(dim_system, dim_reservoir, dtype=torch.double)
        self.W_out = nn.Linear(dim_reservoir, dim_system, dtype=torch.double)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.W_in(x)
        x = self.activation(x)
        x = self.W_out(x)

        return x

def train(train_loader, net, lr, max_epochs, beta=None, stop_loss=None):
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_f = nn.MSELoss()
    for e in range(max_epochs):
        cum_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = net(x)
            loss = loss_f(out, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                cum_loss += loss

                if stop_loss is not None:
                    if cum_loss / (i + 1) < stop_loss:
                        break
        print("Epoch {} mean loss: {}".format(e, cum_loss/(i + 1)))

def generate_trajectory(net, x0, steps):
    x = torch.tensor(x0).unsqueeze(0)

    traj_list = []
    net.eval()
    with torch.no_grad():
        for i in range(steps):
            x = net(x)
            traj_list.append(x[0].cpu().numpy())
    
    traj = np.stack(traj_list)
    return traj

class ChaosDataset(Dataset):
    def __init__(self, train_data):
        self.traj = torch.tensor(train_data)
    
    def __len__(self, ):
        return self.traj.shape[0] - 1

    def __getitem__(self, idx):
        return self.traj[idx], self.traj[idx + 1]

if __name__ == "__main__":
    from data import *
    from visualization import compare, plot_images, short_term_time
    from torch.utils.data import DataLoader

    # dt = 0.02
    # train_data, val_data = get_chen_data(tf=250, dt=dt)
    # train_set = ChaosDataset(train_data)
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    dt = 0.25
    train_data, val_data = KS_from_csv("data/KS_L=44_tf=10000_dt=.25_D=64.csv", 3000, 1000, dt)
    train_set = ChaosDataset(train_data)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    FFNN = FeedForwardNN(64, 3000)

    train(
        train_loader=train_loader,
        net=FFNN,
        lr=0.00001,
        epochs=1000
    )

    predicted = generate_trajectory(
        net=FFNN,
        x0=train_data[-1],
        steps=val_data.shape[0]
    )

    t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])
    #compare(predicted, val_data, t_grid)
    #plot_images(predicted, val_data, 600)
    print(short_term_time(val_data, predicted, dt, tolerance=2e-1))
