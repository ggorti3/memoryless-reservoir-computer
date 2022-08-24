import numpy as np
from reservoir_computer import ReservoirComputer
from neural_net import *
from visualization import short_term_time, compare
from torch.utils.data import DataLoader
from cmrc import CMRC
from torch_data import KS_to_torch
import random

def short_term_test(tf, dt, dim_system, dim_reservoir, data_func, iterations, lambda1):
    #train_data, val_data = data_func(tf=tf, dt=dt)
    train_data, val_data = KS_from_csv("data/KS_L=44_tf=10000_dt=.25_D=64.csv", 8000, 1000, dt)
    train_set = ChaosDataset(train_data)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    MRC_times = []
    RC_times = []
    FFNN_times = []
    for i in range(iterations):
        RC = ReservoirComputer(dim_reservoir=dim_reservoir, dim_system=dim_system, rho=0.9, sigma=0.1, augment=True)
        MRC = MemorylessReservoirComputer(dim_reservoir=dim_reservoir, dim_system=dim_system, sigma=0.1, augment=True)

        # FFNN = FeedForwardNN(dim_system, 2 * dim_reservoir)
        # train(
        #     train_loader=train_loader,
        #     net=FFNN,
        #     lr=0.00001,
        #     epochs=6
        # )

        # predicted3 = generate_trajectory(
        #     net=FFNN,
        #     x0=train_data[-1],
        #     steps=val_data.shape[0]
        # )

        RC.train(train_data)
        predicted2 = RC.predict(val_data.shape[0])

        MRC.train(train_data)
        predicted1 = MRC.predict(val_data.shape[0])

        t1 = short_term_time(val_data, predicted1, dt, tolerance=2e-1)
        t2 = short_term_time(val_data, predicted2, dt, tolerance=2e-1)
        #t3 = short_term_time(val_data, predicted3, dt, tolerance=2e-1)

        MRC_times.append(t1)
        RC_times.append(t2)
        #FFNN_times.append(t3)
    
    MRC_times = np.array(MRC_times)
    RC_times = np.array(RC_times)
    #FFNN_times = np.array(FFNN_times)

    l_time = 1 / lambda1
    MRC_times /= l_time
    RC_times /= l_time
    #FFNN_times /= l_time

    fig = plt.figure()
    ax = fig.gca()

    # ax.plot(["MRC", "RC", "FFNN"], [np.mean(MRC_times), np.mean(RC_times), np.mean(FFNN_times)], "b.")
    # ax.plot(["MRC", "RC", "FFNN"], [np.mean(MRC_times) + np.std(MRC_times), np.mean(RC_times) + np.std(RC_times), np.mean(FFNN_times) + np.std(FFNN_times)], "bx")
    # ax.plot(["MRC", "RC", "FFNN"], [np.mean(MRC_times) - np.std(MRC_times), np.mean(RC_times) - np.std(RC_times), np.mean(FFNN_times) - np.std(FFNN_times)], "bx")

    ax.plot(["MRC", "RC"], [np.mean(MRC_times), np.mean(RC_times)], "b.")
    ax.plot(["MRC", "RC"], [np.mean(MRC_times) + np.std(MRC_times), np.mean(RC_times) + np.std(RC_times)], "bx")
    ax.plot(["MRC", "RC"], [np.mean(MRC_times) - np.std(MRC_times), np.mean(RC_times) - np.std(RC_times)], "bx")

    ax.set_ylabel("Lyapunov Time")
    ax.set_title("KS Short Term Prediction Times")

    print("MRC mean: {}, std_dev:{}".format(np.mean(MRC_times), np.std(MRC_times)))
    print("RC mean: {}, std_dev:{}".format(np.mean(RC_times), np.std(RC_times)))
    #print("FFNN mean: {}, std_dev:{}".format(np.mean(FFNN_times), np.std(FFNN_times)))

    plt.show()

def arbitrary_location_predict(long_traj, tt, tv, dt, dim_system, dim_reservoir, lambda1):
    num_train_steps = int(tt/dt)
    num_val_steps = int(tv/dt)
    rand_idx = random.randrange(long_traj.shape[0] - num_train_steps - num_val_steps)
    train_data = long_traj[:num_train_steps]
    rand_point = long_traj[num_train_steps + rand_idx]
    val_data = long_traj[num_train_steps + 1 + rand_idx:num_train_steps + 1 + rand_idx + num_val_steps]

    RC = ReservoirComputer(dim_reservoir=dim_reservoir, dim_system=dim_system, rho=0.9, sigma=0.1, augment=True)
    MRC = ReservoirComputer(dim_reservoir=dim_reservoir, dim_system=dim_system, rho=0, sigma=0.1, augment=True)
    FFNN = FeedForwardNN(dim_system, 2 * dim_reservoir)
    #FFNN = FeedForwardNN(dim_system, int(0.5 * dim_reservoir))

    train_set = ChaosDataset(train_data)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    train(
        train_loader=train_loader,
        net=FFNN,
        lr=0.0001,
        epochs=600
    )
    predicted3 = generate_trajectory(
        net=FFNN,
        x0=rand_point,
        steps=val_data.shape[0]
    )

    RC.train(train_data)
    RC.advance(rand_point)
    predicted2 = RC.predict(val_data.shape[0])

    MRC.train(train_data)
    MRC.advance(rand_point)
    predicted1 = MRC.predict(val_data.shape[0])

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)

    l_time = 1 / lambda1
    xtick_idxs = np.arange(0, tv, l_time * 2)
    xtick_labels = [2 * i for i in range(xtick_idxs.shape[0])]

    t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])

    ax1.plot(t_grid, val_data[:, 0])
    ax1.plot(t_grid, predicted1[:, 0])
    ax1.set_xticks(xtick_idxs)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_xlabel("x")
    ax1.legend(["truth", "prediction"], loc="lower right")
    ax1.set_title("MRC")
    ax4.plot(t_grid, val_data[:, 1])
    ax4.plot(t_grid, predicted1[:, 1])
    ax4.set_xticks(xtick_idxs)
    ax4.set_xticklabels(xtick_labels)
    ax4.set_xlabel("y")
    ax7.plot(t_grid, val_data[:, 2])
    ax7.plot(t_grid, predicted1[:, 2])
    ax7.set_xticks(xtick_idxs)
    ax7.set_xticklabels(xtick_labels)
    ax4.set_xlabel("z")
    ax7.set_xlabel("Lyapunov Time")

    ax2.plot(t_grid, val_data[:, 0])
    ax2.plot(t_grid, predicted2[:, 0])
    ax2.set_xticks(xtick_idxs)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_title("RC")
    ax5.plot(t_grid, val_data[:, 1])
    ax5.plot(t_grid, predicted2[:, 1])
    ax5.set_xticks(xtick_idxs)
    ax5.set_xticklabels(xtick_labels)
    ax8.plot(t_grid, val_data[:, 2])
    ax8.plot(t_grid, predicted2[:, 2])
    ax8.set_xticks(xtick_idxs)
    ax8.set_xticklabels(xtick_labels)
    ax8.set_xlabel("Lyapunov Time")

    ax3.plot(t_grid, val_data[:, 0])
    ax3.plot(t_grid, predicted3[:, 0])
    ax3.set_xticks(xtick_idxs)
    ax3.set_xticklabels(xtick_labels)
    ax3.set_title("FFNN")
    ax6.plot(t_grid, val_data[:, 1])
    ax6.plot(t_grid, predicted3[:, 1])
    ax6.set_xticks(xtick_idxs)
    ax6.set_xticklabels(xtick_labels)
    ax9.plot(t_grid, val_data[:, 2])
    ax9.plot(t_grid, predicted3[:, 2])
    ax9.set_xticks(xtick_idxs)
    ax9.set_xticklabels(xtick_labels)
    ax9.set_xlabel("Lyapunov Time")

    plt.show()







if __name__ == "__main__":
    from data import *

    # short_term_test(
    #     tf=250,
    #     dt=0.25,
    #     dim_system=64,
    #     dim_reservoir=3000,
    #     data_func=get_chen_data,
    #     iterations=20,
    #     lambda1=0.088
    # )

    long_traj, _ = get_lorenz_data(tf=2000, dt=0.02)
    arbitrary_location_predict(
        long_traj=long_traj,
        tt=250,
        tv=12,
        dt=0.02,
        dim_system=3,
        dim_reservoir=300,
        lambda1=0.902
    )

    from data import *
    from visualization import compare, plot_images, short_term_time
    from torch.utils.data import DataLoader


    # dt = 0.02
    # train_data, val_data = get_lorenz_data(tf=250, dt=dt)
    # train_set = ChaosDataset(train_data)
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    # dt = 0.25
    # train_data, val_data = KS_from_csv("data/KS_L=44_tf=10000_dt=.25_D=64.csv", 3000, 1000, dt)
    # train_set = ChaosDataset(train_data)
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    # FFNN = FeedForwardNN(64, 6000)
    # train(
    #     train_loader=train_loader,
    #     net=FFNN,
    #     lr=0.00001,
    #     epochs=1000
    # )
    # predicted2 = generate_trajectory(
    #     net=FFNN,
    #     x0=train_data[-1],
    #     steps=val_data.shape[0]
    # )
    # np.save("ffnn_pred", predicted2)

    # padding = 8
    # sub_length = 16
    # n = 750
    # training_traj, target = KS_to_torch(train_data, padding, sub_length)
    # cmrc = CMRC(n, sub_length, padding, sigma=0.1)
    # cmrc.generate_r_states(training_traj)
    # cmrc.train_normal_eq(target)
    # assisted_vec = np.concatenate([train_data[-1, val_data.shape[1] - padding:], train_data[-1], train_data[-1, :padding]])
    # assisted_vec = torch.tensor(assisted_vec, dtype=torch.double)
    # cmrc.advance(assisted_vec)
    # predicted3 = cmrc.predict(val_data.shape[0])
    # np.save("cmrc_pred", predicted3)


    # MRC = ReservoirComputer(dim_reservoir=3000, dim_system=64, rho=0, sigma=0.1, augment=True)
    # MRC.train(train_data)
    # predicted1 = MRC.predict(val_data.shape[0])
    # np.save("mrc_pred", predicted1)

    #t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])
    #compare(predicted, val_data, t_grid)
    #plot_images(predicted, val_data, 600)

    ####predicted, actual, num_preds, dt=0.25, lyapunov=0.088
    # num_preds = 300
    # predicted2 = np.load("ffnn_pred.npy")[:num_preds]
    # #predicted3 = np.load("cmrc_pred.npy")[:num_preds]
    # predicted3 = predicted3[:num_preds]
    # predicted1 = np.load("mrc_pred.npy")[:num_preds]

    # lyapunov=0.088
    # d = predicted1.shape[1]
    # n = min(num_preds, predicted1.shape[0])
    # l_time = 1 / lyapunov
    # l_steps = l_time / dt
    # xtick_idxs = np.arange(0, n, l_steps)
    # xtick_labels = [i for i in range(xtick_idxs.shape[0])]

    # ytick_idxs = np.arange(0, 64 + 01e-4, 64//2)
    # ytick_labels = range(0, 44 + 1, 44//2)

    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)
    # vmin, vmax = -3, 3

    # ax1.imshow(val_data.transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    # ax1.set_title("Truth")
    # ax1.set_xticks(xtick_idxs)
    # ax1.set_xticklabels(xtick_labels)
    # ax1.set_yticks(ytick_idxs)
    # ax1.set_yticklabels(ytick_labels)
    # ax2.imshow(val_data.transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    # ax2.set_title("Truth")
    # ax2.set_xticks(xtick_idxs)
    # ax2.set_xticklabels(xtick_labels)
    # ax2.set_yticks(ytick_idxs)
    # ax2.set_yticklabels(ytick_labels)
    # ax3.imshow(val_data.transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    # ax3.set_title("Truth")
    # ax3.set_xticks(xtick_idxs)
    # ax3.set_xticklabels(xtick_labels)
    # ax3.set_yticks(ytick_idxs)
    # ax3.set_yticklabels(ytick_labels)

    # ax4.imshow(predicted3.transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    # ax4.set_title("Conv. MRC Prediction")
    # ax4.set_xticks(xtick_idxs)
    # ax4.set_xticklabels(xtick_labels)
    # ax4.set_yticks(ytick_idxs)
    # ax4.set_yticklabels(ytick_labels)
    # ax7.imshow((val_data[:num_preds] - predicted3[:num_preds]).transpose(), cmap='coolwarm', vmin=vmin, vmax=vmax)
    # ax7.set_title("Conv. MRC Difference")
    # ax7.set_xticks(xtick_idxs)
    # ax7.set_xticklabels(xtick_labels)
    # ax7.set_yticks(ytick_idxs)
    # ax7.set_yticklabels(ytick_labels)
    # ax7.set_xlabel("Lyapunov Time")

    # ax5.imshow(predicted1.transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    # ax5.set_title("MRC Prediction")
    # ax5.set_xticks(xtick_idxs)
    # ax5.set_xticklabels(xtick_labels)
    # ax5.set_yticks(ytick_idxs)
    # ax5.set_yticklabels(ytick_labels)
    # ax8.imshow((val_data[:num_preds] - predicted1[:num_preds]).transpose(), cmap='coolwarm', vmin=vmin, vmax=vmax)
    # ax8.set_title("MRC Difference")
    # ax8.set_xticks(xtick_idxs)
    # ax8.set_xticklabels(xtick_labels)
    # ax8.set_yticks(ytick_idxs)
    # ax8.set_yticklabels(ytick_labels)
    # ax8.set_xlabel("Lyapunov Time")

    # ax6.imshow(predicted2.transpose()[:, :num_preds], cmap='coolwarm', vmin=vmin, vmax=vmax)
    # ax6.set_title("FFNN Prediction")
    # ax6.set_xticks(xtick_idxs)
    # ax6.set_xticklabels(xtick_labels)
    # ax6.set_yticks(ytick_idxs)
    # ax6.set_yticklabels(ytick_labels)
    # ax9.imshow((val_data[:num_preds] - predicted2[:num_preds]).transpose(), cmap='coolwarm', vmin=vmin, vmax=vmax)
    # ax9.set_title("FFNN Difference")
    # ax9.set_xticks(xtick_idxs)
    # ax9.set_xticklabels(xtick_labels)
    # ax9.set_yticks(ytick_idxs)
    # ax9.set_yticklabels(ytick_labels)
    # ax9.set_xlabel("Lyapunov Time")

    # plt.show()

