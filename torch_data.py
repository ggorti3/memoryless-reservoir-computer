import numpy as np
import torch
from torch.utils.data import Dataset



def KS_to_torch(train_data, padding, sub_length):
    # sub length must divide train_data.shape[1]
    training_traj = torch.tensor(train_data, dtype=torch.double)
    if padding > 0:
        training_traj = torch.cat([training_traj[:, training_traj.shape[1] - padding:], training_traj, training_traj[:, :padding]], dim=1)
    training_traj = training_traj.unsqueeze(1)
    target = torch.tensor(train_data, dtype=torch.double).reshape((-1, sub_length))
    return training_traj, target

def KS_from_csv(data_path, tt, tv, dt):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    
    state_vec_list = []
    for line in lines:
        state_vec_list.append(np.fromstring(line, dtype=np.double, sep=','))
    traj = np.stack(state_vec_list)
    train_steps = int(tt/dt)
    val_steps = int(tv/dt)
    train_traj = traj[:train_steps]
    val_traj = traj[train_steps:train_steps + val_steps]
    return train_traj, val_traj

class KS_Dataset(Dataset):
    def __init__(self, traj, padding, sub_length):
        """
        traj is np array
        """
        self.sub_length = sub_length
        self.padding = padding
        self.training_traj, _ = KS_to_torch(traj, padding, sub_length)
        self.last_vec = self.training_traj[-1]
        self.num_windows = (self.training_traj.shape[2] - 2 * self.padding) // sub_length
    
    def __len__(self,):
        return self.training_traj.shape[0] - 1
    
    def __getitem__(self, idx):
        return self.training_traj[idx], self.training_traj[idx + 1, :, self.padding:self.training_traj.shape[2] - self.padding]
    

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    padding = 8
    sub_length = 16
    dt = 0.25

    train_data, val_data = KS_from_csv("data/ks_tf=10000_dt=0.25_L=60_Q=128.csv", 800, 200, 0.25)
    training_traj, target = KS_to_torch(train_data, padding, sub_length)
    torch_dset = KS_Dataset(train_data, padding, sub_length)
    dataloader = DataLoader(torch_dset, batch_size=400, shuffle=True)

    batch, label = next(iter(dataloader))
    print(batch.shape)
    print(label.shape)
    print(torch_dset.last_vec.shape)