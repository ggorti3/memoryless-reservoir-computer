import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from utils import correlation_distance
from scipy.signal import argrelmin, argrelmax

### functions for plotting results ###

def short_term_time(actual, predicted, dt, tolerance=1e-2):
    _, d = actual.shape
    i = 0
    while True:
        x1 = actual[i]
        x2 = predicted[i]
        error = (np.sum((x2 - x1)**2)**0.5) / (np.sum(x1**2)**0.5)
        if error > tolerance:
            return dt * i
        i += 1

def compare(predicted, actual, t, fontsize = 10):
    """
    Plot a comparison between a predicted trajectory and actual trajectory.
    
    Plots up to 9 dimensions
    """
    dimensions = actual.shape[1]
    plt.clf()
    plt.ion()
    
    i = 0
    while i < min(dimensions, 9):
        if i == 0:
            var = "x"
        elif i == 1:
            var = "y"
        elif i == 2:
            var = "z"
        else:
            var = ("dimension {}" .format((i + 1)))
        
        plt.subplot(min(dimensions, 9), 1, (i + 1))
        plt.plot(t, actual[:, i])
        plt.plot(t, predicted[:, i])
        plt.ylabel(var, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if i == 0:
            plt.legend(["truth", "predicted"])
            plt.title("Prediction vs Truth")
        i += 1
        
    plt.xlabel("Time", fontsize=fontsize)
    plt.show()
    input("Press enter to exit")
    
def plot_poincare(predicted):
    """
    Displays the poincare plot of the given predicted trajectory
    """
    plt.clf()
    plt.ion()   
    
    zp = predicted[:, 2]
    
    zpmaxes = zp[argrelextrema(zp, np.greater)[0]]
    zpi = zpmaxes[0:(zpmaxes.shape[0] - 1)]
    zpi1 = zpmaxes[1:]
    
    plt.scatter(zpi, zpi1)
    plt.xlabel("z_i")
    plt.ylabel("z_(i + 1)")
    plt.title("Poincare Plot")
    plt.show()
    input("Press enter to exit")

def plot_images(predicted, actual, num_preds, dt=0.25, lyapunov=0.088, L=44):

    d = predicted.shape[1]
    n = min(num_preds, predicted.shape[0])
    l_time = 1 / lyapunov
    l_steps = l_time / dt
    xtick_idxs = np.arange(0, n, l_steps)
    xtick_labels = [i for i in range(xtick_idxs.shape[0])]

    ytick_idxs = np.arange(0, d + 01e-4, d//2)
    ytick_labels = range(0, L + 1, L//2)


    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    vmin, vmax = -3, 3
    ax1.imshow(actual.transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax1.set_title("Truth")
    ax1.set_xticks(xtick_idxs)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_yticks(ytick_idxs)
    ax1.set_yticklabels(ytick_labels)
    ax2.imshow(predicted.transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax2.set_title("Prediction")
    ax2.set_xticks(xtick_idxs)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_yticks(ytick_idxs)
    ax2.set_yticklabels(ytick_labels)
    ax3.imshow((actual - predicted).transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax3.set_title("Difference")
    ax3.set_xticks(xtick_idxs)
    ax3.set_xticklabels(xtick_labels)
    ax3.set_yticks(ytick_idxs)
    ax3.set_yticklabels(ytick_labels)
    ax3.set_xlabel("Lyapunov Time")
    plt.show()

def plot_correlations(traj):
    num_gridpoints = traj.shape[1]
    my_range = np.array(range(num_gridpoints))
    c_dists = []
    for i in range(num_gridpoints):
        c_vec = correlation_distance(traj, i)
        c_dists.append(np.mean(c_vec))

    c_dists = np.array(c_dists)

    max_idxs = argrelmax(c_dists[:c_dists.shape[0]//2 + 1])[0]
    min_idxs = argrelmin(c_dists[:c_dists.shape[0]//2 + 1])[0]
    extrema = np.sort(np.concatenate([np.array([1]), c_dists[max_idxs], np.absolute(c_dists[min_idxs])]))[::-1]
    extrema_idxs = np.sort(np.concatenate([np.array([0]), my_range[max_idxs], my_range[min_idxs]]))

    coeffs = np.polyfit(extrema_idxs, np.log(extrema), 1)
    print(coeffs)
    envelope = np.exp(coeffs[0] * my_range)

    fig, ax = plt.subplots(1)
    ax.plot(my_range, c_dists)
    ax.plot(my_range, envelope)
    ax.set_title("correlation at each distance")
    ax.set_xticks(range(0, num_gridpoints + 1, 10))
    plt.show()

def plot_3d(truth, predicted):
    pass

if __name__ == "__main__":
    sample_size = 20
    def data_from_txt(save_path, sample_size):
        with open(save_path, 'r') as f:
            lines = f.readlines()
        RC_exps_strs = lines[2:2+sample_size]
        MRC_exps_strs = lines[4+sample_size:]

        RC_exps_list = []
        MRC_exps_list = []
        for i in range(sample_size):

            RC_exps_str = RC_exps_strs[i]
            RC_exps_list.append(np.fromstring(RC_exps_str, sep=","))

            MRC_exps_str = MRC_exps_strs[i]
            MRC_exps_list.append(np.fromstring(MRC_exps_str, sep=","))
        
        RC_exps = np.stack(RC_exps_list)
        MRC_exps = np.stack(MRC_exps_list)

        return RC_exps, MRC_exps
    
    import math

    digit_dict = {
        0:"\u2080",
        1:"\u2081",
        2:"\u2082",
        3:"\u2083",
        4:"\u2084",
        5:"\u2085",
        6:"\u2086",
        7:"\u2087",
        8:"\u2088",
        9:"\u2089"
    }


    lorenz_exps = [ 9.01943283e-01, -4.92518729e-05, -1.45479136e+01]
    chen_exps = [ 1.45007078e-01, -2.76087019e-04, -5.51935099e+00]
    rossler_exps = [ 7.21984003e-02,  3.10838739e-06, -5.38831374e+00]
    dadras_exps = [ 3.78572652e-01, -7.54695521e-04, -1.06718951e+01]

    ks_exps = [0.088, 0.054, 0.032, 0.011, 0.002, -0.001, -0.002, -0.005, -0.018, -0.060, -0.120, -0.178, -0.245, -0.297, -0.335, -0.370, -0.418, -0.479, -0.720, -0.728]
    ks_exps_adjusted = [0.088, 0.054, 0.032, 0.011, 0.002, -0.005, -0.018, -0.060, -0.120, -0.178, -0.245, -0.297, -0.335, -0.370, -0.418, -0.479, -0.720, -0.728]
    ks_exps_mrc = [0.089, 0.064, 0.029, 0.020, 0.011, -0.010, -0.039, -0.058, -0.138, -0.170, -0.253, -0.300, -0.323, -0.356, -0.390, -0.455]
    ks_exps_rc = [0.107, 0.078, 0.043, 0.020, 0.003, -0.005, -0.042, -0.068, -0.109, -0.176, -0.239, -0.279, -0.349, -0.403, -0.448, -0.461]
    ks_exps_nn = [0.147, 0.105, 0.077, 0.037, 0.005, -0.020, -0.046, -0.077, -0.105, -0.156, -0.195, -0.242, -0.304, -0.362, -0.463, -0.541, -0.890, -1.117, -1.500, -1.790]
    ks_exps_cmrc = [0.091, 0.069, 0.053, 0.034, 0.020, 0.006, -0.018, -0.042, -0.068, -0.110, -0.185, -0.239, -0.287, -0.339, -0.373, -0.406, -0.492, -0.690, -0.848, -1.124]
    num_exps = 16

    fig = plt.figure()
    ax = fig.gca()

    # x_vals = ["\u03BB{}".format(i + 1) for i in range(num_exps)]
    # ax.plot(x_vals, ks_exps[:num_exps], "x", markersize=8)
    # ax.plot(x_vals, ks_exps_cmrc[:num_exps], ".", markersize=8)
    # ax.plot(x_vals, ks_exps_mrc[:num_exps], ".", markersize=8)
    # ax.plot(x_vals, ks_exps_rc[:num_exps], ".", markersize=8)
    # ax.plot(x_vals, ks_exps_nn[:num_exps], ".", markersize=8)
    

    # ax.legend(["truth", "CMRC", "MRC", "RC", "FFNN"])
    # ax.set_ylim([-0.6, 0.16])

    cmrc_error = np.abs(np.array(ks_exps_cmrc[:num_exps]) - np.array(ks_exps[:num_exps]))
    mrc_error = np.abs(np.array(ks_exps_mrc[:num_exps]) - np.array(ks_exps[:num_exps]))
    rc_error = np.abs(np.array(ks_exps_rc[:num_exps]) - np.array(ks_exps[:num_exps]))
    nn_error = np.abs(np.array(ks_exps_nn[:num_exps]) - np.array(ks_exps[:num_exps]))

    # plot_dist = 8
    # width = 1
    # cmrc_x = np.arange(0, plot_dist * num_exps, plot_dist)
    # mrc_x = np.arange(1, plot_dist * num_exps, plot_dist)
    # rc_x = np.arange(2, plot_dist * num_exps, plot_dist)
    # nn_x = np.arange(3, plot_dist * num_exps, plot_dist)

    # ax.bar(cmrc_x, cmrc_error, width=width, align='edge')
    # ax.bar(mrc_x, mrc_error, width=width, align='edge')
    # ax.bar(rc_x, rc_error, width=width, align='edge')
    # ax.bar(nn_x, nn_error, width=width, align='edge')

    # ax.set_xticks(np.arange(2, plot_dist * num_exps, plot_dist))
    # ax.set_xticklabels(np.array(["\u03BB{}".format(i + 1) for i in range(num_exps)]))

    # ax.legend(["CMRC", "MRC", "RC", "FFNN"])

    x_vals = ["\u03BB{}".format(i + 1) for i in range(num_exps)]

    base_str = "\u03BB"
    x_vals = []
    for i in range(num_exps):
        x_str = base_str
        num_digits = int(math.log10(i + 1)) + 1
        for j in reversed(range(num_digits)):
            digit = ((i + 1) // (10 ** j)) % 10
            x_str +=  digit_dict[digit]
        x_vals.append(x_str)
    print(x_vals)

    marker = ".-"

    ax.plot(x_vals, cmrc_error, marker)
    ax.plot(x_vals, mrc_error, marker)
    ax.plot(x_vals, rc_error, marker)
    ax.plot(x_vals, nn_error, marker)

    ax.legend(["CMRC", "MRC", "RC", "FFNN"])
    ax.set_ylabel("Absolute Error")


    plt.show()
    
