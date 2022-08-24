import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import lin_reg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}".format(DEVICE))

class CMRC(nn.Module):
    def __init__(self, n, sub_length, padding, sigma=0.1):
        # we can model the operation of the subsampling RLC using a convolution along space at one point in time
        # filters should be size (sub_length, )
        # number of filters = number of nodes in hidden state
        # stride should be size of prediction window (sub_length - 2 * padding)
        # then, another conv layer will map the output of the first conv layer to the next system state
        super(CMRC, self).__init__()
        self.n = n
        self.sub_length = sub_length
        self.padding = padding

        W_in = 2 * sigma * (np.random.rand(n, sub_length + 2 * padding) - 0.5)
        self.conv = nn.Conv1d(1, n, sub_length + 2 * padding, stride=sub_length, bias=False).type(torch.double)
        self.conv.weight.data = torch.tensor(W_in, dtype=torch.double).unsqueeze(1)
        self.conv2 = nn.Conv1d(2 * n, sub_length, 1, stride=1, bias=False).type(torch.double)
        self.sig = nn.Sigmoid()
    
    def generate_r_states(self, traj):
        """
        generate reservoir states for each state vector in training trajectory

        :params:
        traj (torch.tensor): training trajectory (num_samples, 1, len_state_vector + 2 * padding)
        """
        with torch.no_grad():
            num_samples = traj.shape[0]
            r_states = self.conv(traj) # out is (num_samples, n, len_state_vector // sub_length)
            r_states = self.activation(r_states)
            self.r_state = r_states[-1].unsqueeze(0)

            num_subwindows = r_states.shape[2]
            r_states = r_states.transpose(1, 2).reshape((num_samples * num_subwindows, 2 * self.n))
            self.training_states = torch.cat([torch.zeros((num_subwindows, 2 * self.n)), r_states[:r_states.shape[0] - num_subwindows]], dim=0)
    
    def train_normal_eq(self, target):
        """
        using generated reservoir states, find W_out with normal equations

        :params:
        traj (torch.tensor): training trajectory (num_samples, 1, len_state_vector + 2 * padding)
        """
        U = target.cpu().numpy()
        R = self.training_states.cpu().numpy().transpose()
        W_out = lin_reg(R, U, 0.0001)
        self.conv2.weight.data = torch.tensor(W_out, dtype=torch.double).unsqueeze(2)

    def readout(self):
        with torch.no_grad():
            pred = self.conv2(self.r_state).transpose(1, 2).flatten()
            if self.padding > 0:
                assisted_vec = torch.cat([pred[pred.shape[0] - self.padding:], pred, pred[:self.padding]])
            else:
                assisted_vec = pred
            
            return pred, assisted_vec

    def advance(self, assisted_vec):
        with torch.no_grad():
            out = self.conv(assisted_vec.unsqueeze(0).unsqueeze(0))
            self.r_state = self.activation(out)
    
    def activation(self, x):
        x = self.sig(x)
        x = torch.cat([x, x**2], dim=1)
        return x

    def predict(self, steps):
        pred_list = []
        for i in range(steps):
            pred, assisted_vec = self.readout()
            pred_list.append(pred.detach().cpu().numpy())
            self.advance(assisted_vec)
        predicted = np.stack(pred_list)
        return predicted

class CNN(nn.Module):
    def __init__(self, n, sub_length, padding):
        super(CNN, self).__init__()
        self.n = n
        self.sub_length = sub_length
        self.padding = padding

        self.conv = nn.Conv1d(1, n, sub_length + 2 * padding, stride=sub_length).type(torch.double)
        #self.conv.requires_grad = False
        self.conv2 = nn.Conv1d(2 * n, sub_length, 1, stride=1).type(torch.double)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.activation(x)
        x = torch.cat([x, x**2], dim=1)
        x = self.conv2(x)
        x = self.activation(x).transpose(1, 2).reshape(batch_size, 1 ,  -1)
        return x
    
    def predict(self, u0, steps):
        pred_list = []
        assisted_vec = u0
        for i in range(steps):
            pred = model(assisted_vec)
            assisted_vec = torch.cat([pred[:, :, :self.padding], pred, pred[:, :, pred.shape[2] - self.padding:]], axis=2)
            pred_list.append(pred.detach().cpu().flatten().numpy())
        predicted = np.stack(pred_list)
        return predicted


def train(model, dataloader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.5)
    loss_f = nn.MSELoss()
    model.train()
    epoch_loss_list = []
    fig, ax = plt.subplots(1)
    for e in range(epochs):
        cum_loss = 0
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_f(out, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                cum_loss += loss
            # if i % 10 == 0 and i != 0:
            #     scheduler.step()
        if e % 50 == 0:
            print("Epoch {} complete".format(e))
            epoch_loss_list.append((cum_loss / i).item())

            ax.clear()
            ax.plot(epoch_loss_list)
            plt.savefig("GD_training_loss.png")



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch_data import KS_from_csv, KS_to_torch, KS_Dataset
    from torch.utils.data import DataLoader
    from visualization import plot_images, plot_correlations, compare, short_term_time
    from reservoir_computer import ReservoirComputer

    padding = 8
    sub_length = 16
    n = 750
    dt = 0.25

    train_data, val_data = KS_from_csv("data/KS_L=200_tf=10000_dt=.25_D=256.csv", 3000, 1000, 0.25)

    training_traj, target = KS_to_torch(train_data, padding, sub_length)
    print("data generated.")

    print("instantiating model...")
    model = CMRC(n, sub_length, padding, sigma=0.1)

    model.to(DEVICE)
    print("generating training r_states...")
    model.generate_r_states(training_traj)

    print("fitting model...")
    model.train_normal_eq(target)


    assisted_vec = np.concatenate([train_data[-1, val_data.shape[1] - padding:], train_data[-1], train_data[-1, :padding]])
    assisted_vec = torch.tensor(assisted_vec, dtype=torch.double)
    model.advance(assisted_vec)
    predicted = model.predict(val_data.shape[0])

    plot_images(predicted, val_data, 500, dt=dt, lyapunov=0.089)

    
    
