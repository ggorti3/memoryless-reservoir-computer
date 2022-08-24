import numpy as np
import torch
import torch.nn as nn

from utils import generate_reservoir, scale_res, lin_reg, sigmoid

# This file contains the ReservoirComputer class and necessary helper functions
# The main method at the bottom contains a demo of the code, run this file in the terminal to check it out

class ReservoirComputer(MemorylessReservoirComputer):
    def __init__(self, dim_system=3, dim_reservoir=300, sigma=0.1, rho=1.1, density=0.05, augment=True):
        super().__init__(dim_system=dim_system, dim_reservoir=dim_reservoir, sigma=sigma, augment=augment)
        self.A = generate_reservoir(dim_reservoir, rho, density)
    
    def advance(self, u):
        """
        Generate the next r_state given an input u and the current r_state
        """
        
        self.r_state = 2 * (sigmoid(np.dot(self.A, self.r_state) + np.dot(self.W_in, u)) - 0.5)

class MemorylessReservoirComputer:
    def __init__(self, dim_system=3, dim_reservoir=300, sigma=0.1, augment=True):
        self.dim_reservoir = dim_reservoir
        self.dim_system = dim_system
        self.augment = augment
        self.r_state = np.zeros(dim_reservoir)
        self.W_in = 2 * sigma * (np.random.rand(dim_reservoir, dim_system) - 0.5)
        if self.augment:
            self.W_out = np.zeros((dim_system, 2 * dim_reservoir))
        else:
            self.W_out = np.zeros((dim_system, dim_reservoir))
    
    def advance(self, u):
        """
        Generate the next r_state given an input u and the current r_state
        """
        
        self.r_state = 2 * (sigmoid(np.dot(self.W_in, u)) - 0.5)
        
    def readout(self):
        """
        Generate and return the prediction v given the current r_state
        """
        if self.augment:
            r_temp = np.concatenate([self.r_state, self.r_state**2])
            v = np.dot(self.W_out, r_temp)
        else:
            v = np.dot(self.W_out, self.r_state)
        return v
    
    def train(self, traj):
        """
        Optimize W_out so that the network can accurately model the given trajectory.
        
        Parameters
        traj: The training trajectory stored as a (n, dim_system) dimensional numpy array, where n is the number of timesteps in the trajectory
        """
        R = np.zeros((self.dim_reservoir, traj.shape[0]))
        for i in range(traj.shape[0]):
            R[:, i] = self.r_state
            x = traj[i]
            self.advance(x)
        if self.augment:
            R = np.concatenate([R, R** 2])
        self.W_out = lin_reg(R, traj, 0.0001)
        print("train loss: {}".format(np.mean((self.W_out @ R - traj.transpose())**2)))
        
    def predict(self, steps):
        """
        Use the network to generate a series of predictions
        
        Parameters
        steps: the number of predictions to make. Can be any positive integer
        
        Returns
        predicted: the predicted trajectory stored as a (steps, dim_system) dimensional numpy array
        """
        predicted = np.zeros((steps, self.dim_system))
        for i in range(steps):
            v = self.readout()
            predicted[i] = v
            self.advance(v)
        return predicted

def compare_times(data_path, train_times, iterations):
    MRC_avg_times = []
    RC_avg_times = []
    dt=0.25
    for train_time in train_times:
        train_data, val_data = KS_from_csv("data/KS_L=44_tf=10000_dt=.25_D=64.csv", train_time, 500, dt)
        MRC_times = []
        RC_times = []
        print("Time {}".format(train_time))
        for i in range(iterations):
            MRC = ReservoirComputer(dim_reservoir=2000, dim_system=64, rho=0, sigma=0.1, augment=True)
            MRC.train(train_data)
            predicted1 = MRC.predict(val_data.shape[0])
            MRC_time = short_term_time(predicted1, val_data, dt, tolerance=2e-1)
            MRC_times.append(MRC_time)

            RC = ReservoirComputer(dim_reservoir=2000, dim_system=64, rho=0.9, sigma=0.1, augment=True)
            RC.train(train_data)
            predicted2 = RC.predict(val_data.shape[0])
            RC_time = short_term_time(predicted2, val_data, dt, tolerance=2e-1)
            RC_times.append(RC_time)
            print("    Iteration {}".format(i))

        MRC_avg_times.append(np.mean(np.array(MRC_times)).item())
        RC_avg_times.append(np.mean(np.array(RC_times)).item())
    print(MRC_avg_times)
    print(RC_avg_times)



if __name__ == "__main__":
    from data import *
    from visualization import compare, plot_poincare, plot_images, short_term_time
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    dt = 0.02
    # train_data, val_data = get_lorenz_data(tf=250, dt=dt)
    # train_data, val_data = get_KS_data(num_gridpoints=100, tf=5000, dt=dt)
    # train_data, val_data = KS_from_csv("data/KS_L=44_tf=10000_dt=.25_D=64.csv", 3000, 1000, dt)
    train_data, val_data = get_chen_data(tf=1000, dt=dt, skip=25, split=0.8)
    #train_data, val_data = get_rossler_data(tf=800, dt=dt, skip=25, split=0.8)
    # train_data, val_data = get_dadras_data(tf=400, dt=dt, skip=25, split=0.8)

    MRC = ReservoirComputer(dim_reservoir=300, dim_system=3, rho=0, sigma=0.01, augment=True)
    MRC.train(train_data)
    predicted = MRC.predict(val_data.shape[0])



    # RC = ReservoirComputer(dim_reservoir=3000, dim_system=64, rho=0, sigma=0.1, augment=True)
    # RC.train(train_data)
    # predicted = RC.predict(val_data.shape[0])

    # plot_images(predicted, val_data, 600)


    t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])
    compare(predicted, val_data, t_grid)


    # compare_times(
    #     data_path="data/KS_L=44_tf=10000_dt=.25_D=64.csv",
    #     train_times=[500, 1000, 2000, 4000, 6000, 8000],
    #     iterations=20)

    # training_times = ["500", "1000", "2000", "4000", "6000"]
    # MRC_times =  [3.85, 6.925, 7.2, 9.325, 13.6625]
    # RC_times =  [3.825, 6.525, 7.1625, 9.4125, 14.1375]

    # fig = plt.figure()
    # ax = plt.subplot()
    # ax.plot(training_times, MRC_times)
    # ax.plot(training_times, RC_times)
    # ax.set_xlabel("Training Time")
    # ax.set_ylabel("Accurate Short Term Prediction Time")
    # ax.legend(["MRC", "RC"])
    # plt.show()
