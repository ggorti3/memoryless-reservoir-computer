import numpy as np
import scipy
from reservoir_computer import ReservoirComputer
from utils import sigmoid
from visualization import plot_poincare
from data import RK4
import matplotlib.pyplot as plt

from neural_net import *

import matlab
import matlab.engine

def lyapunov_exponents(x0, integrator_f, dt, T, k):
    d = x0.shape[0]
    Q = np.eye(d, dtype=np.double)
    vec0 = np.concatenate([x0, Q.flatten()], dtype=np.double)
    R_diag_list = []
    for i in range(k):
        ts, traj = RK4(integrator_f, vec0, T, dt)
        x_traj = traj[:, :d]
        x = traj[-1, :d]
        phi = traj[-1, d:].reshape(d, d)
        Q, R = scipy.linalg.qr(phi)
        R_diag_list.append(np.diag(R))

        vec0 = np.concatenate([x, Q.flatten()])
    R_diag_arr = np.stack(R_diag_list)
    lyapunov_exps = np.mean(np.log(np.abs(R_diag_arr)), axis=0) / T
    print(lyapunov_exps)
    return lyapunov_exps

def lorenz_F(x, t, sigma, rho, beta):
    x_prime = np.zeros(x.shape)
    
    x_prime[0] = sigma * (x[1] - x[0])
    x_prime[1] = x[0] * (rho - x[2]) - x[1]
    x_prime[2] = x[0] * x[1] - beta * x[2]
    return x_prime

def lorenz_jacobian(u, sigma, rho, beta):
    x, y, z = u[0], u[1], u[2]
    J = np.array(
        [
            [-sigma, sigma, 0],
            [rho - z, -1, -x],
            [y, x, -beta]
        ]
    )
    return J

def chen_F(r, t, a, b, c):
    x = r[0]; y = r[1]; z = r[2]
    u = (a * x) - (y * z)
    v = (b * y) + (x * z)
    w = (c * z) + (x * y/3)
    return np.array([u, v, w])

def chen_jacobian(u, a, b, c):
    x, y, z = u[0], u[1], u[2]
    J = np.array(
        [
            [a, -z, -y],
            [z, b, x],
            [y/3, x/3, c]
        ]
    )
    return J

def rossler_F(r, t, a, b, c):
    x = r[0]; y = r[1]; z = r[2]
    u = - y - z
    v = x + a * y
    w = b + z * (x-c)
    return np.array([u, v, w])

def rossler_jacobian(u, a, b, c):
    x, y, z = u[0], u[1], u[2]
    J = np.array(
        [
            [0, -1, -1],
            [1, a, 0],
            [z, 0, x - c]
        ]
    )
    return J

def dadras_F(r, t, a, b, c, d, e):
    x = r[0]; y = r[1]; z = r[2]
    u = y - (a * x) + (b * y * z)
    v = (c * y) - (x * z) + z
    w = (d * x * y) - (e * z)
    return np.array([u, v, w])

def dadras_jacobian(u, a, b, c, d, e):
    x, y, z = u[0], u[1], u[2]
    J = np.array(
        [
            [-a, 1 + b * z, b * y],
            [-z, c, 1 - x],
            [d * y, d * x, -e]
        ]
    )
    return J



class VariationalEquation():
    def __init__(self, d, F, jacobian, kw_coeffs):
        """
        d: dimension of system
        F: the vector valued function that gives first derivative at any point x
        jacobian: function that gives jacobian at any point x
        """
        self.d = d
        self.kw_coeffs = kw_coeffs
        self.F = F
        self.jacobian = jacobian
    
    def var_F(self, vec, t):
        """
        integrates the variational equation
        """
        x = vec[:self.d]
        x_prime = self.F(x, t, **self.kw_coeffs)

        phi = vec[self.d:].reshape(self.d, self.d)
        phi_prime = self.jacobian(x, **self.kw_coeffs) @ phi

        return np.concatenate([x_prime, phi_prime.flatten()])

    

def lyapunov_exponents_discrete(x0, T, dt, k, net):
    d = x0.shape[0]
    n = net.dim_reservoir
    x = x0
    Q = np.eye(d)
    R_diag_list = []
    pred_list = [x]
    t_steps = int(T / dt)
    for i in range(k):
        #J = np.eye(d)
        J = Q
        J2 = np.zeros((n, d))
        for j in range(t_steps):
            x, J, J2 = RC_jacobian(x, J, J2, net)
            pred_list.append(x)
        #phi = J @ Q
        Q, R = scipy.linalg.qr(J)
        R_diag_list.append(np.diag(R))
    R_diag_arr = np.stack(R_diag_list)
    predicted = np.stack(pred_list)
    #plot_poincare(predicted)
    lyapunov_exps = np.mean(np.log(np.abs(R_diag_arr)), axis=0) / T
    print(lyapunov_exps)
    return lyapunov_exps

def lyapunov_exponents_discrete2(x0, T, dt, k, net):
    d = x0.shape[0]
    n = net.dim_reservoir
    x = x0
    Q = np.eye(d)
    R_diag_list = []
    pred_list = [x]
    t_steps = int(T / dt)

    J_in = (net.r_state * (1 - net.r_state))[:, np.newaxis] * net.W_in
    Q, R = scipy.linalg.qr(J_in)
    R_diag_list.append(np.diag(R))
    
    for i in range(k):
        J = np.eye(net.dim_reservoir)
        for j in range(t_steps):
            x, J_part = reservoir_jacobian(x, net)
            J = J_part @ J
            pred_list.append(x)
        Q, R = scipy.linalg.qr(J @ Q)
        R_diag_list.append(np.diag(R)[:d])

        if i % 10 == 0:
            print(i)

    net.advance(x)
    J_out = net.W_out[:, :net.dim_reservoir] + 2 * net.r_state[np.newaxis, :] * net.W_out[:, net.dim_reservoir:]
    Q, R = scipy.linalg.qr(J_out @ Q)
    R_diag_list.append(np.diag(R))

    R_diag_arr = np.stack(R_diag_list)
    predicted = np.stack(pred_list)
    #plot_poincare(predicted)
    lyapunov_exps = np.sum(np.log(np.abs(R_diag_arr)), axis=0) / (T*k)
    print(lyapunov_exps)
    return lyapunov_exps

def RC_jacobian(x, J, J2, net):
    #h = net.r_state.copy()
    net.advance(x)
    J_out = net.W_out[:, :net.dim_reservoir] + 2 * net.r_state[np.newaxis, :] * net.W_out[:, net.dim_reservoir:]

    #sig_term = sigmoid(net.W_in @ x + net.A @ h)
    sig_term = net.r_state
    J1_part = (sig_term * (1 - sig_term))[:, np.newaxis] * net.W_in
    J2_part = (sig_term * (1 - sig_term))[:, np.newaxis] * net.A

    J2 = J1_part @ J + J2_part @ J2
    J = J_out @ J2

    return net.readout(), J, J2

def reservoir_jacobian(x, net):
    J2 = net.W_out[:, :net.dim_reservoir] + 2 * net.r_state[np.newaxis, :] * net.W_out[:, net.dim_reservoir:]
    net.advance(x)

    sig_term = (net.r_state + 1) / 2
    J1 = 2 * (sig_term * (1 - sig_term))[:, np.newaxis] * net.A
    J3 = 2 * (sig_term * (1 - sig_term))[:, np.newaxis] * net.W_in
    J = J1 + J3 @ J2

    return net.readout(), J


def lyapunov_exponents_ks(x0, dt, T, k, perturb=1e-12):
    d = x0.shape[0]
    Q = np.eye(d, dtype=np.double)
    R_diag_list = []
    eng = matlab.engine.start_matlab()
    for i in range(k):
        # estimate J using the perturb with the matlab integrator, integrating for time T with steps dt
        u0_list = []
        for p in range(d):
            u0_list.append([x0[p]])
        u0 = matlab.double(u0_list)
        traj = eng.evolve_KS(u0, T, dt, "full")
        traj = np.array(traj).transpose()
        #_, traj = generateLorenz(x0, T, dt, 10, 28, 8/3)

        phi_cols = []
        for j in range(d):
            pert_vec = perturb * Q[:, j]
            x0_pert = x0 + pert_vec

            u0_pert_list = []
            for p in range(d):
                u0_pert_list.append([x0_pert[p]])
            u0_pert = matlab.double(u0_pert_list)
            pert_traj = eng.evolve_KS(u0_pert, T, dt, "full")
            pert_traj = np.array(pert_traj).transpose()
            #_, pert_traj = generateLorenz(x0_pert, T, dt, 10, 28, 8/3)

            phi_cols.append((pert_traj[-1] - traj[-1]) / perturb)
        phi = np.stack(phi_cols, axis=1)

        x0 = traj[-1]
        Q, R = scipy.linalg.qr(phi)
        R_diag_list.append(np.diag(R))

        print("iteration {} complete".format(i))

    eng.quit()
    R_diag_arr = np.stack(R_diag_list)
    lyapunov_exps = np.mean(np.log(np.abs(R_diag_arr)), axis=0) / T
    print(lyapunov_exps)
    return lyapunov_exps

def lyapunov_exponents_experimental_nn(x0, dt, T, k, net, perturb=1e-12):
    d = x0.shape[0]
    Q = np.eye(d, dtype=np.double)
    R_diag_list = []
    for i in range(k):
        # estimate J using the perturb with the matlab integrator, integrating for time T with steps dt
        traj = generate_trajectory(
            net=net,
            x0=x0,
            steps= int(T / dt)
        )

        phi_cols = []
        for j in range(d):
            pert_vec = perturb * Q[:, j]
            x0_pert = x0 + pert_vec

            pert_traj = generate_trajectory(
                net=net,
                x0=x0_pert,
                steps=int(T / dt)
            )

            phi_col = (pert_traj[-1] - traj[-1]) / perturb
            phi_cols.append(phi_col)

        x0 = traj[-1]
        phi = np.stack(phi_cols, axis=1)
        Q, R = scipy.linalg.qr(phi)
        R_diag_list.append(np.diag(R))
        print("iteration {} complete".format(i))

    R_diag_arr = np.stack(R_diag_list)
    lyapunov_exps = np.mean(np.log(np.abs(R_diag_arr)), axis=0) / T
    print(lyapunov_exps)
    return lyapunov_exps

def lyapunov_exponents_experimental_rc(x0, dt, T, k, net, perturb=1e-12):
    d = x0.shape[0]
    Q = np.eye(d, dtype=np.double)
    R_diag_list = []
    for i in range(k):
        # estimate J using the perturb with the rc, integrating for time T with steps dt
        r_state = net.r_state
        net.advance(x0)
        traj = net.predict(int(T / dt))
        new_r_state = net.r_state

        phi_cols = []
        for j in range(d):
            pert_vec = perturb * Q[:, j]
            x0_pert = x0 + pert_vec

            net.r_state = r_state
            net.advance(x0_pert)
            pert_traj = net.predict(int(T / dt))

            phi_col = (pert_traj[-1] - traj[-1]) / perturb
            phi_cols.append(phi_col)

        net.r_state = new_r_state
        x0 = traj[-1]
        phi = np.stack(phi_cols, axis=1)
        Q, R = scipy.linalg.qr(phi)
        R_diag_list.append(np.diag(R))
        print("iteration {} complete".format(i))

    R_diag_arr = np.stack(R_diag_list)
    lyapunov_exps = np.mean(np.log(np.abs(R_diag_arr)), axis=0) / T
    print(lyapunov_exps)
    return lyapunov_exps

def lyapunov_exponents_experimental_cmrc(x0, dt, T, k, net, perturb=1e-12):
    d = x0.shape[0]
    Q = np.eye(d, dtype=np.double)
    R_diag_list = []
    for i in range(k):
        # estimate J using the perturb with the rc, integrating for time T with steps dt
        assisted_vec = np.concatenate([x0[x0.shape[0] - net.padding:], x0, x0[:net.padding]])
        assisted_vec = torch.tensor(assisted_vec, dtype=torch.double)
        net.advance(assisted_vec)
        traj = net.predict(int(T / dt))

        phi_cols = []
        for j in range(d):
            pert_vec = perturb * Q[:, j]
            x0_pert = x0 + pert_vec

            assisted_vec_pert = np.concatenate([x0_pert[x0.shape[0] - net.padding:], x0_pert, x0_pert[:net.padding]])
            assisted_vec_pert = torch.tensor(assisted_vec_pert, dtype=torch.double)
            net.advance(assisted_vec_pert)
            pert_traj = net.predict(int(T / dt))

            phi_col = (pert_traj[-1] - traj[-1]) / perturb
            phi_cols.append(phi_col)

        x0 = traj[-1]
        phi = np.stack(phi_cols, axis=1)
        Q, R = scipy.linalg.qr(phi)
        R_diag_list.append(np.diag(R))
        print("iteration {} complete".format(i))

    R_diag_arr = np.stack(R_diag_list)
    lyapunov_exps = np.mean(np.log(np.abs(R_diag_arr)), axis=0) / T
    print(lyapunov_exps)
    return lyapunov_exps

def search_T(start, stop, step, x0, dt, k, net=None):
    d = x0.shape[0]
    exps_list = [[] for i in range(d)]
    for T in np.arange(start, stop + 1e-6, step):
        # exps = lyapunov_exponents_ks(
        #     x0=x0,
        #     dt=dt,
        #     T=T.item(),
        #     k=k
        # )
        exps = lyapunov_exponents_experimental_cmrc(
            x0=x0,
            dt=dt,
            T=T.item(),
            k=k,
            net=net
        )
        for i, e in enumerate(exps):
            exps_list[i].append(e)
    
    
    T_range = np.arange(start, stop + 1e-6, step)
    num_exps = min(20, d)
    header_string = "T   |"
    for i in range(num_exps):
        header_string += " " + "exp {}".format(i + 1).ljust(6) + " |"
    print(header_string)

    for i, T in enumerate(T_range):
        info_string = "{:.2f}".format(T).ljust(4) + "|"
        for j in range(num_exps):
            info_string += " " + "{:.3f}".format(exps_list[j][i]).ljust(6) + " |"
        print(info_string)

    for i in range(num_exps):
        fig = plt.figure()
        ax = plt.gca()
        #T_range = np.arange(start, stop + 1e-6, step)
        #for i in range(len(exps_list)):
        ax.plot(T_range, exps_list[i], ".-")
        ax.set_title("exp {}".format(i + 1))
    plt.show()



if __name__ == "__main__":
    from data import *
    import random
    from torch.utils.data import *
    from visualization import compare, plot_images
    from torch_data import KS_from_csv, KS_to_torch, KS_Dataset
    from cmrc import CMRC



    train_data, val_data = KS_from_csv("data/KS_L=44_tf=10000_dt=.25_D=64.csv", 3000, 500, 0.25)

    padding=8
    training_traj, target = KS_to_torch(train_data, padding, 16)
    model = CMRC(750, 16, padding, sigma=0.1)
    print("generating training r_states...")
    model.generate_r_states(training_traj)
    print("fitting model...")
    model.train_normal_eq(target)
    assisted_vec = np.concatenate([train_data[-1, val_data.shape[1] - padding:], train_data[-1], train_data[-1, :padding]])
    assisted_vec = torch.tensor(assisted_vec, dtype=torch.double)
    model.advance(assisted_vec)
    predicted = model.predict(val_data.shape[0])
    plot_images(predicted, val_data, 500, lyapunov=0.088)

    # train_data, val_data = get_lorenz_data(tf=250, dt=0.02)

    # train_data, val_data = KS_from_csv("data/KS_L=200_tf=10000_dt=.25_D=256.csv", 3000, 1000, 0.25)
    # x0 = train_data[-1]
    # lyapunov_exponents_ks(
    #     x0=x0,
    #     dt=0.25,
    #     T=3,
    #     k=1000,
    #     perturb=1e-12
    # )

    x0 = train_data[-1]
    search_T(1.25, 3, 0.25, x0, 0.25, 1000, model)

    #dt = 0.25
    #train_data, val_data = get_rossler_data(tf=250, dt=dt)
    # train_set = ChaosDataset(train_data)
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    # FFNN = FeedForwardNN(64, 3000)

    # train(
    #     train_loader=train_loader,
    #     net=FFNN,
    #     lr=0.00001,
    #     epochs=1000
    # )

    # predicted = generate_trajectory(
    #     net=FFNN,
    #     x0=train_data[-1],
    #     steps=val_data.shape[0]
    # )

    # RC = ReservoirComputer(dim_reservoir=3000, dim_system=64, rho=0.9, sigma=0.1, augment=True)
    # RC.train(train_data)
    # r_state = RC.r_state
    # predicted = RC.predict(val_data.shape[0])
    # plot_images(predicted, val_data, 600)
    # RC.r_state = r_state

    #t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])
    #compare(predicted, val_data, t_grid)

    # x0 = train_data[-1]
    # lyapunov_exponents_experimental_cmrc(
    #     x0=x0,
    #     dt=0.25,
    #     T=3,
    #     k=2000,
    #     net=model,
    #     perturb=1e-12
    # )
    #search_T(1.25, 3, 0.25, x0, 0.25, 1000, RC)
    # lyapunov_exponents_discrete2(
    #     x0=x0,
    #     T=2,
    #     dt=0.25,
    #     k=200,
    #     net=RC)
    #plot_images(predicted, val_data, 600)

    #t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])
    #compare(predicted, val_data, t_grid)

    # predicted = generate_trajectory(
    #     net=FFNN,
    #     x0=train_data[-1],
    #     steps=val_data.shape[0]
    # )

    # t_grid = np.linspace(0, val_data.shape[0] * dt, val_data.shape[0])
    # compare(predicted, val_data, t_grid)


    # name = "KS"
    # T = 2
    # k = 100
    # #sample_size = 1
    # training_times = [500, 750, 1000, 2000, 5000, 10000]
    # #training_times = [500, 750]
    # #tf = 1300
    # dt = 0.25
    # rho=0.9
    # sigma=0.1
    # dim_reservoir = 2000

    # RC_exps_list = []
    # MRC_exps_list = []
    # for tf in training_times:
    #     train_data, val_data = KS_from_csv("data/KS_L=44_tf=10000_dt=.25_D=64.csv", tf, 500, dt)
    #     RC = ReservoirComputer(dim_reservoir=dim_reservoir, dim_system=64, rho=rho, sigma=sigma, augment=True)
    #     RC.train(train_data)
    #     x0 = RC.readout()

    #     RC_exps = lyapunov_exponents_discrete2(
    #         x0=x0,
    #         T=T,
    #         dt=dt,
    #         k=k,
    #         net=RC
    #     )
    #     RC_exps_list.append(RC_exps)

    #     MRC = ReservoirComputer(dim_reservoir=dim_reservoir, dim_system=64, rho=0, sigma=sigma, augment=True)
    #     MRC.train(train_data)
    #     x0 = MRC.readout()

    #     MRC_exps = lyapunov_exponents_discrete2(
    #         x0=x0,
    #         T=T,
    #         dt=dt,
    #         k=k,
    #         net=MRC
    #     )
    #     MRC_exps_list.append(MRC_exps)
    
    # save_path = "out/{}_results.txt".format(name)
    # with open(save_path, 'w') as f:
    #     f.write("dt={}, T={}, k={}, tf={}, sample_size={}, rho={}, sigma={}, dim_reservoir={}\n".format(dt, T, k, tf, sample_size, rho, sigma, dim_reservoir))

    #     f.write("RC exps: \n")
    #     for exps in RC_exps_list:
    #         f.write(str(exps.tolist())[1:-1] + "\n")
    #     f.write("\n")

    #     f.write("MRC exps: \n")
    #     for exps in MRC_exps_list:
    #         f.write(str(exps.tolist())[1:-1] + "\n")
    #     f.write("\n")

    # lorenz_exps = [ 9.01943283e-01, -4.92518729e-05, -1.45479136e+01]
    # chen_exps = [ 1.45007078e-01, -2.76087019e-04, -5.51935099e+00]
    # rossler_exps = [ 7.21984003e-02,  3.10838739e-06, -5.38831374e+00]
    # dadras_exps = [ 3.78572652e-01, -7.54695521e-04, -1.06718951e+01]


    



