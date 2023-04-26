import numpy as np
import networkx as nx
import random

def generate_reservoir(dim_reservoir, rho, density):
    """
    Generates a random reservoir matrix with the desired rho and density
    
    Returns the reservoir matrix A as a (dim_reservoir, dim_reservoir) dimensional numpy array
    """
    graph = nx.gnp_random_graph(dim_reservoir, density)
    array = nx.to_numpy_array(graph)
    rand = 2 * (np.random.rand(dim_reservoir) - 0.5)
    res = array * rand
    return scale_res(res, rho)

def scale_res(A, rho):
    """
    Scales the given reservoir matrix A such that its spectral radius is rho.
    """
    eigvalues, eigvectors = np.linalg.eig(A)
    max_eig = np.amax(eigvalues)
    max_length = np.absolute(max_eig)
    if max_length == 0:
        raise ZeroDivisionError("Max of reservoir eigenvalue lengths cannot be zero.")
    return rho * A / max_length

def lin_reg(R, U, beta=0.0001):
    """
    Return an optimized matrix using ridge regression.

    Parameters
    R: The generated reservoir states, stored as a (dim_system, n) dimensional numpy array
    U: The training trajectory, stored as a (n, dim_system) dimensional numpy array
    beta: regularization parameter
    
    Returns
    W_out: the optimized W_out array
    """
    
    Rt = np.transpose(R)
    W_out = np.matmul(np.matmul(np.transpose(U), Rt), np.linalg.inv(np.matmul(R, Rt) + beta * np.identity(R.shape[0])))

    return W_out

def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def relu(x):
    return np.where(x >= 0, x, 0)

def correlation_distance(traj, x_idx):
    num_gridpoints = traj.shape[1]
    temp = np.tile(traj, (num_gridpoints, 1, 1))
    temp2 = np.transpose(traj)
    u_t_temp = temp2 - np.mean(temp, axis=2)
    var_temp = np.sum(u_t_temp * u_t_temp, axis=0)

    x_idx = x_idx % num_gridpoints
    if x_idx == 0:
        u_tau_temp = u_t_temp
    else:
        u_tau_temp = np.concatenate([u_t_temp[x_idx:], u_t_temp[:x_idx]], axis=0)
    
    covar_temp = np.sum(u_t_temp * u_tau_temp, axis=0)

    correlation_distance_vec = covar_temp / var_temp
    return correlation_distance_vec
