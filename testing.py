import numpy as np
from lyapunov import lyapunov_exponents_experimental_rc, lyapunov_exponents_experimental_nn, lyapunov_exponents_experimental_cmrc
from reservoir_computer import ReservoirComputer, MemorylessReservoirComputer
from neural_net import *
from visualization import short_term_time, compare
from torch.utils.data import DataLoader
from cmrc import CMRC
from torch_data import KS_to_torch
import random

def rsfn_grid_search(traj, n_train, n_val, dim_reservoirs, deltas, densities, betas, iterations=20, k=20):
    mean_times = []
    j = 0
    n = len(dim_reservoirs) * len(deltas) * len(densities) * len(betas)
    for dim_reservoir in dim_reservoirs:
        for delta in deltas:
            for density in densities:
                for beta in betas:
                    times = []
                    for i in range(iterations):
                        idx = random.randrange(traj.shape[0] - n_train - n_val)
                        training_data = traj[idx:(idx + n_train)]
                        val_data = traj[(idx + n_train):(idx + n_train + n_val)]

                        rsfn = MemorylessReservoirComputer(
                            dim_system=traj.shape[1],
                            dim_reservoir=dim_reservoir,
                            delta=delta,
                            in_density=density,
                            beta=beta
                        )

                        rsfn.train(training_data)
                        pred = rsfn.predict(n_val)
                        time = short_term_time(val_data, pred, 0.02, tolerance=2e-1)
                        times.append(time)
                    
                    mean_time = np.mean(np.array(times))
                    print("config {}/{}".format(j + 1, n))
                    j += 1
                    mean_times.append(mean_time)
    
    mean_times = np.array(mean_times)
    sort_idxs = np.argsort(mean_times)[::-1]

    print("dim_reservoir   delta   density   beta        | mean_time   ")
    print("____________________________________________________________")
    for i in sort_idxs[:k]:
        dim_reservoir = dim_reservoirs[(i // (len(deltas) * len(densities) * len(betas)))]
        delta = deltas[(i // (len(densities) * len(betas))) % len(deltas)]
        density = densities[(i // len(betas)) % len(densities)]
        beta = betas[i % len(betas)]
        mean_time = mean_times[i]
        print("{:<16}{:<8.4f}{:<10.4f}{:<12.2E}| {:<12.4f}".format(
            dim_reservoir,
            delta,
            density,
            beta,
            mean_time
        ))

def rc_grid_search(traj, n_train, n_val, dim_reservoirs, deltas, in_densities, rhos, densities, betas, iterations=20, k=20):
    mean_times = []
    j = 0
    n = len(dim_reservoirs) * len(deltas) * len(in_densities) * len(rhos) * len(densities) * len(betas)
    for dim_reservoir in dim_reservoirs:
        for delta in deltas:
            for in_density in in_densities:
                for rho in rhos:
                    for density in densities:
                        for beta in betas:
                            times = []
                            for i in range(iterations):
                                idx = random.randrange(traj.shape[0] - n_train - n_val)
                                training_data = traj[idx:(idx + n_train)]
                                val_data = traj[(idx + n_train):(idx + n_train + n_val)]

                                rc = ReservoirComputer(
                                    dim_system=traj.shape[1],
                                    dim_reservoir=dim_reservoir,
                                    delta=delta,
                                    in_density=in_density,
                                    rho=rho,
                                    density=density,
                                    beta=beta
                                )

                                rc.train(training_data)
                                pred = rc.predict(n_val)
                                time = short_term_time(val_data, pred, 0.02, tolerance=2e-1)
                                times.append(time)
                            
                            mean_time = np.mean(np.array(times))
                            print("config {}/{}".format(j + 1, n))
                            j += 1
                            mean_times.append(mean_time)
    
    mean_times = np.array(mean_times)
    sort_idxs = np.argsort(mean_times)[::-1]

    print("dim_reservoir   delta   in_density   rho   density   beta        | mean_time   ")
    print("_______________________________________________________________________________")
    for i in sort_idxs[:k]:
        dim_reservoir = dim_reservoirs[(i // (len(deltas) * len(in_densities) * len(rhos) * len(densities) * len(betas)))]
        delta = deltas[(i // (len(in_densities) * len(rhos) * len(densities) * len(betas))) % len(deltas)]
        in_density = in_densities[(i // (len(rhos) * len(densities) * len(betas))) % len(in_densities)]
        rho = rhos[(i // (len(densities) * len(betas))) % len(rhos)]
        density = densities[(i // len(betas)) % len(densities)]
        beta = betas[i % len(betas)]
        mean_time = mean_times[i]
        print("{:<16}{:<8.4f}{:<13.4f}{:<6.3f}{:<10.4f}{:<12.2E}| {:<12.4f}".format(
            dim_reservoir,
            delta,
            in_density,
            rho,
            density,
            beta,
            mean_time
        ))

def nn_grid_search(traj, n_train, n_val, dim_hiddens, stop_losses, betas, batch_size, lr, max_epochs, iterations=20, k=20):
    mean_times = []
    j = 0
    n = len(dim_hiddens) * len(stop_losses) * len(betas)
    for dim_hidden in dim_hiddens:
        for stop_loss in stop_losses:
            for beta in betas:
                times = []
                for i in range(iterations):
                    idx = random.randrange(traj.shape[0] - n_train - n_val)
                    training_data = traj[idx:(idx + n_train)]
                    val_data = traj[(idx + n_train):(idx + n_train + n_val)]

                    train_set = ChaosDataset(training_data)
                    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

                    FFNN = FeedForwardNN(val_data.shape[1], dim_hidden)

                    train(
                        train_loader=train_loader,
                        net=FFNN,
                        lr=lr * (dim_hidden / 300),
                        max_epochs=max_epochs,
                        beta=beta,
                        stop_loss=stop_loss,
                        hide=True
                    )

                    pred = generate_trajectory(
                        net=FFNN,
                        x0=training_data[-1],
                        steps=val_data.shape[0]
                    )

                    time = short_term_time(val_data, pred, 0.02, tolerance=2e-1)
                    times.append(time)
                
                mean_time = np.mean(np.array(times))
                print("config {}/{}".format(j + 1, n))
                j += 1
                mean_times.append(mean_time)
    
    mean_times = np.array(mean_times)
    sort_idxs = np.argsort(mean_times)[::-1]

    print("dim_hiddens   stop_loss   beta        | mean_time   ")
    print("____________________________________________________________")
    for i in sort_idxs[:k]:
        dim_hidden = dim_hiddens[(i // (len(stop_losses) * len(betas)))]
        stop_loss = stop_losses[(i // len(betas)) % len(stop_losses)]
        beta = betas[i % len(betas)]
        mean_time = mean_times[i]
        print("{:<14}{:<12.4f}{:<12.4E}| {:<12.4f}".format(
            dim_hidden,
            stop_loss,
            beta,
            mean_time
        ))

def short_term_test(tf, dt, rc_params, mrc_params, nn_params, data_func, iterations, lambda1):
    x0 = np.random.rand(3)
    train_data, val_data = data_func(tf=tf, dt=dt, x0=x0)
    #train_data, val_data = KS_from_csv("data/KS_L=44_tf=10000_dt=.25_D=64.csv", 8000, 1000, dt)
    train_set = ChaosDataset(train_data)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    MRC_times = []
    RC_times = []
    FFNN_times = []
    for i in range(iterations):
        RC = ReservoirComputer(**rc_params)
        MRC = MemorylessReservoirComputer(**mrc_params)

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


def rc_climate_test(tf, dt, model_class, model_params, data_func, T, k, iterations):
    x0 = np.random.rand(3)
    train_data, val_data = data_func(tf=tf, dt=dt, x0=x0)
    x0 = train_data[-1]

    exps_list = []
    for i in range(iterations):
        model = model_class(**model_params)
        model.train(train_data)
        
        exps = lyapunov_exponents_experimental_rc(
            x0=x0,
            dt=dt,
            T=T,
            k=k,
            net=model,
            perturb=1e-12
        )
        exps_list.append(exps)
    
    all_exps = np.stack(exps_list)
    mean_exps = np.mean(all_exps, axis=0)
    std_exps = np.std(all_exps, axis=0)
    print("All Exps:")
    print(all_exps)
    print("Mean Exps:")
    print(mean_exps)
    print("Std Exps")
    print(std_exps)


if __name__ == "__main__":
    from data import *
    from torch.utils.data import DataLoader

    traj, _ = get_chen_data(tf=12500, dt=0.02)

    dim_reservoirs = [300]
    deltas = np.exp(np.linspace(np.log(0.01), np.log(3), 15))[4:12]
    in_densities = [0.5, 0.75, 1]
    rhos = [0.8, 0.9, 1, 1.1, 1.2]
    densities = [0.05, 0.1, 0.2, 0.5, 0.75, 1]

    betas = np.exp(np.linspace(np.log(1e-6), np.log(0.01), 4))
    stop_losses = [0] + np.exp(np.linspace(np.log(0.0004), np.log(0.1), 6)).tolist()
    dim_hiddens = [50, 150, 300]
    dim_hiddens = [300]

    # dim_reservoirs = [300]
    # deltas = np.exp(np.linspace(np.log(0.01), np.log(3), 3))
    # in_densities = [1]
    # rhos = [0.9, 1.1]
    # densities = [0.05, 0.2]
    # betas = np.exp(np.linspace(np.log(1e-6), np.log(0.1), 3))


    # rsfn_grid_search(
    #     traj=traj,
    #     n_train=10000,
    #     n_val=2500,
    #     dim_reservoirs=dim_reservoirs,
    #     deltas=deltas,
    #     densities=in_densities,
    #     betas=betas,
    #     iterations=20
    # )

    # rc_grid_search(
    #     traj=traj,
    #     n_train=10000,
    #     n_val=2500,
    #     dim_reservoirs=dim_reservoirs,
    #     deltas=deltas,
    #     in_densities=in_densities,
    #     rhos=rhos,
    #     densities=densities,
    #     betas=betas,
    #     iterations=20,
    # )

    nn_grid_search(
        traj=traj,
        n_train=10000,
        n_val=2500,
        dim_hiddens=dim_hiddens,
        stop_losses=stop_losses,
        betas=betas,
        batch_size=2500,
        lr=1e-2,
        max_epochs=200,
        iterations=20
    )

    rc_params_lorenz = {
        "dim_system":3,
        "dim_reservoir":300,
        "delta":0.0510,
        "in_density":1,
        "rho":1.2,
        "density":0.5,
        "beta":1e-6
    }

    mrc_params_lorenz = {
        "dim_system":3,
        "dim_reservoir":300,
        "delta":0.0510,
        "in_density":1,
        "beta":1e-6
    }

    nn_params_lorenz = {
        "dim_system":3,
        "dim_reservoir":300,
        "stop_loss":0.0110,
        "beta":2.1544e-5
    }

    rc_params_dadras = {
        "dim_system":3,
        "dim_reservoir":300,
        "delta":0.1732,
        "in_density":1,
        "rho":0.8,
        "density":0.5,
        "beta":1e-6
    }

    mrc_params_dadras = {
        "dim_system":3,
        "dim_reservoir":300,
        "delta":0.1152,
        "in_density":1,
        "beta":1e-6
    }

    nn_params_dadras = {
        "dim_system":3,
        "dim_reservoir":300,
        "stop_loss":0.0004,
        "beta":2.1544e-5
    }

    rc_params_rossler = {
        "dim_system":3,
        "dim_reservoir":300,
        "delta":0.0510,
        "in_density":1,
        "rho":0.9,
        "density":0.75,
        "beta":1e-6
    }

    mrc_params_rossler = {
        "dim_system":3,
        "dim_reservoir":300,
        "delta":0.0510,
        "in_density":1,
        "beta":1e-6
    }

    nn_params_rossler = {
        "dim_system":3,
        "dim_reservoir":300,
        "stop_loss":0,
        "beta":4.6416e-4
    }

    rc_params_chen = {
        "dim_system":3,
        "dim_reservoir":300,
        "delta":0.2603,
        "in_density":1,
        "rho":1.2,
        "density":0.5,
        "beta":1.78e-5
    }

    mrc_params_chen = {
        "dim_system":3,
        "dim_reservoir":300,
        "delta":0.3912,
        "in_density":1,
        "beta":1.78e-5
    }

    nn_params_chen = {
        "dim_system":3,
        "dim_reservoir":300,
        "stop_loss":0.0004,
        "beta":1e-6
    }


    # short_term_test(
    #     tf=250,
    #     dt=0.02,
    #     rc_params=rc_params,
    #     mrc_params=mrc_params,
    #     nn_params=nn_params,
    #     data_func=get_dadras_data,
    #     iterations=20,
    #     lambda1=0.38
    # )

    # train_data, val_data = get_dadras_data(tf=250, dt=0.02)
    # x0 = train_data[-1]
    # rsbn = MemorylessReservoirComputer(**mrc_params_dadras)
    # rsbn = MemorylessReservoirComputer(dim_system=3, dim_reservoir=300, delta=0.1, in_density=0.2, beta=0.0001)
    # rsbn.train(train_data)
    # predicted = rsfn.predict(val_data.shape[0])

    # exps = lyapunov_exponents_experimental_rc(
    #     x0=x0,
    #     dt=0.02,
    #     T=10,
    #     k=1000,
    #     net=rsbn,
    #     perturb=1e-12
    # )
    # print(exps)

    # rc_climate_test(
    #     tf=250,
    #     dt=0.02,
    #     model_class=MemorylessReservoirComputer,
    #     model_params=mrc_params_dadras,
    #     data_func=get_dadras_data,
    #     T=10,
    #     k=2000,
    #     iterations=20
    # )

    # model_configs = [(MemorylessReservoirComputer, mrc_params), (ReservoirComputer, rc_params)]
    # data_funcs = [get_lorenz_data, get_rossler_data, get_chen_data, get_dadras_data]


    # for model_class, model_params in model_configs:
    #     for data_func in data_funcs:
    #         print(model_class)
    #         print(data_func)
    #         print("T=10")
    #         rc_climate_test(
    #             tf=250,
    #             dt=0.02,
    #             model_class=model_class,
    #             model_params=model_params,
    #             data_func=data_func,
    #             T=10,
    #             k=2000,
    #             iterations=20
    #         )
    #         print("T=0.2")
    #         rc_climate_test(
    #             tf=250,
    #             dt=0.02,
    #             model_class=model_class,
    #             model_params=model_params,
    #             data_func=data_func,
    #             T=0.2,
    #             k=2000,
    #             iterations=20
    #         )
    #         print()
    