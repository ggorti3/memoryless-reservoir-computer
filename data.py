import numpy as np
import random
import matplotlib.pyplot as plt

def RK4(f, r0, tf, dt):
    """Fourth-order Runge-Kutta integrator.

    :param f: Function to be integrated
    :param r0: Initial conditions
    :param tf: Integration duration
    :param dt: Timestep size
    :returns: time and trajectory vectors

    """
    
    # generate an array of time steps
    ts = np.arange(0, tf, dt)
    
    # create an array to hold system state at each timestep
    traj = np.zeros((ts.shape[0], len(r0)))
    traj[0, :] = np.array(r0)
    
    # calculate system state at each time step, save it in the array
    for i in range(0, ts.shape[0]-1):
        t = ts[i]
        r = traj[i, :]

        k1 = dt * f(r, t)
        k2 = dt * f(r + k1/2, t + dt/2)
        k3 = dt * f(r + k2/2, t + dt/2)
        k4 = dt * f(r + k3, t + dt)
        K = (1.0/6)*(k1 + 2*k2 + 2*k3 + k4)

        traj[i+1, :] = r + K
    
    return (ts, traj)

def generateLorenz(r0, tf, dt, sigma, rho, beta):
    """Integrate a given Lorenz system."""

    # define equations of lorenz system
    def lorenz(r, t):
        x = r[0]; y = r[1]; z = r[2]
        u = sigma * (y - x)
        v = x * (rho - z) - y
        w = x * y - beta * z
        return np.array([u, v, w])

    ts, traj = RK4(lorenz, r0, tf, dt)
    return (ts, traj)

def generateChen(r0, tf, dt, a, b, c):
    """Integrate a given Chen system."""

    # define equations of Chen system
    def chen(r, t):
        x = r[0]; y = r[1]; z = r[2]
        u = (a * x) - (y * z)
        v = (b * y) + (x * z)
        w = (c * z) + (x * y/3)
        return np.array([u, v, w])

    ts, traj = RK4(chen, r0, tf, dt)
    return (ts, traj)

def get_chen_data(tf=400, dt=0.02, skip=25, split=0.8):
    _, traj = generateChen((1, 1, 1), tf, dt, 5, -10, -.38)
    
    skip_steps = int(25 / dt)
    traj = traj[skip_steps:]
    
    split_num = int(split * traj.shape[0])
    
    train_data = traj[:split_num]
    val_data = traj[split_num:]
    
    return train_data, val_data

def generateDadras(r0, tf, dt, a, b, c, d, e):
    """Integrate a given Dadras system."""

    # define equations of Dadras system
    def dadras(r, t):
        x = r[0]; y = r[1]; z = r[2]
        u = y - (a * x) + (b * y * z)
        v = (c * y) - (x * z) + z
        w = (d * x * y) - (e * z)
        return np.array([u, v, w])

    ts, traj = RK4(dadras, r0, tf, dt)
    return (ts, traj)

def get_dadras_data(tf=100, dt=0.02, skip=25, split=0.8):
    _, traj = generateDadras((1, 1, 1), tf, dt, 3, 2.7, 1.7, 2, 9)
    
    skip_steps = int(25 / dt)
    traj = traj[skip_steps:]
    
    split_num = int(split * traj.shape[0])
    
    train_data = traj[:split_num]
    val_data = traj[split_num:]
    
    return train_data, val_data

def generateRossler(r0, tf, dt, a, b, c):
    """Integrate a given Rossler system."""

    # define equations of Rossler system
    def rossler(r, t):
        x = r[0]; y = r[1]; z = r[2]
        u = - y - z
        v = x + a * y
        w = b + z * (x-c)
        return np.array([u, v, w])

    ts, traj = RK4(rossler, r0, tf, dt)
    return (ts, traj)

def get_rossler_data(tf=800, dt=0.01, skip=25, split=0.8):
    _, traj = generateRossler((1, 1, 1), tf, dt, .2, .2, 5.7)
    
    skip_steps = int(25 / dt)
    traj = traj[skip_steps:]
    
    split_num = int(split * traj.shape[0])
    
    train_data = traj[:split_num]
    val_data = traj[split_num:]
    
    return train_data, val_data

def get_lorenz_data(tf=250, dt=0.02, skip=25, split=0.8):
    _, traj = generateLorenz((1, 1, 1), tf, dt, 10, 28, 8/3)
    
    skip_steps = int(25 / dt)
    traj = traj[skip_steps:]
    
    split_num = int(split * traj.shape[0])
    
    train_data = traj[:split_num]
    val_data = traj[split_num:]
    
    return train_data, val_data

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

def dist(x, y):
    diff = x - y
    ax = len(diff.shape) - 1
    return np.sum(diff**2, axis=ax)**0.5
