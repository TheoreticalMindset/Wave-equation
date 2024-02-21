import numpy as np
import scipy.sparse.linalg as spsplg
import scipy.linalg as splg
from scipy.sparse import kron, csc_matrix, eye
import matplotlib.pyplot as plt
import operators as ops
import rungekutta4 as rk4
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_simulation(mx=200, my=100, order=6, show_animation=True):
    # Model parameters
    c = 1  # wave speed
    T = 3  # end time
    k = 2 * np.pi  # k
    w_ang = c * k  # angular velocity
    xl, xr = -1, 1  # left and right boundaries
    yl, yr = -1/2, 1/2  # top and bottom boundaries
    L = xr - xl  # domain length

    # Space discretization
    hx = (xr - xl) / mx
    hy = (yr - yl) / my
    xvec, hx = np.linspace(xl, xr, mx, retstep=True)
    yvec, hy = np.linspace(yl, yr, my, retstep=True)
    x_grid, y_grid = np.meshgrid(xvec, yvec, indexing='ij')

    #phi0 = xvec + yve
    

    def create_combined_vector(mx, my, v0):
        # Create V matrix
        V = np.zeros((my, mx))
        for j in range(len(xvec)):
            for i in range(len(yvec)):
                #print(yvec[i])
                V[i, j] = initial(xvec[j], yvec[i])

        T_vector = V.flatten(order='F')
        zero_vector = np.zeros_like(T_vector)
        combined_vector = np.concatenate((T_vector, zero_vector))

        return combined_vector


    v0 = 2
    w = create_combined_vector(mx, my, v0)
    #print(w.shape)  # Output: (20000,)


    if order == 2:
        Hx, HIx, D1x, D2x, e_lx, e_rx, d1_lx, d1_rx = ops.sbp_cent_2nd(mx, hx)
        Hy, HIy, D1y, D2y, e_ly, e_ry, d1_ly, d1_ry = ops.sbp_cent_2nd(my, hy)
    elif order == 4:
        Hx, HIx, D1x, D2x, e_lx, e_rx, d1_lx, d1_rx = ops.sbp_cent_4th(mx, hx)
        Hy, HIy, D1y, D2y, e_ly, e_ry, d1_ly, d1_ry = ops.sbp_cent_4th(my, hy)
    elif order == 6:
        Hx, HIx, D1x, D2x, e_lx, e_rx, d1_lx, d1_rx = ops.sbp_cent_6th(mx, hx)
        Hy, HIy, D1y, D2y, e_ly, e_ry, d1_ly, d1_ry = ops.sbp_cent_6th(my, hy)

    Imx = eye(mx)
    Imy = eye(my)

    # D_mx = (c**2)*(D2x + HIx@e_lx@d1_lx.T - HIx@e_rx@d1_rx.T)
    # D_my = (c**2)*(D2y + HIy@e_ry@d1_ry.T - HIy@e_ly@d1_ly.T)

    D_mx = (c**2)*(D2x + HIx@e_lx@d1_lx.T - HIx@e_rx@d1_rx.T)
    D_my = (c**2)*(D2y + HIy@e_ly@d1_ly.T - HIy@e_ry@d1_ry.T)

    D = kron(Imy, D_mx) + kron(D_my, Imx) # fixed
    D = csc_matrix(D)  # Convert to sparse format

    # Uncomment to check eigenvalues. Will tell you if the solution is stable
    # eigD = spsplg.eigvals(D.toarray())
    # plt.figure()
    # plt.scatter(eigD.real, eigD.imag)
    # plt.show()

    # Define right-hand-side function
    def rhs(w):
        phi, phi_t = np.split(w, 2)
        phi_tt = D @ phi
        return np.concatenate([phi_t, phi_tt]) # t, tt

    # Time discretization
    ht_try = 0.1 * min(hx, hy) / c
    mt = int(np.ceil(T / ht_try) + 1)
    tvec, ht = np.linspace(0, T, mt, retstep=True)

    # Initialize time variable and solution vector
    t = 0
    #u = np.split(w, 2)
    l = len(w) // 2
    u = w[:l]

    #print(xvec.shape, yvec.shape, u.shape)

    # Initialize plot for animation
    if show_animation:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def update_plot(tidx):
            ax.clear()  # Clear the previous plot
            surface = ax.plot_surface(x_grid, y_grid, u.reshape((mx, my)), cmap='viridis')
            ax.set_xlim([xl, xr-hx])
            ax.set_ylim([yl, yr-yl])
            ax.set_zlim([-0.6, 0.6])
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('u')
            ax.set_title(f't={round(t, 2)}')
            plt.draw()
            plt.pause(1e-8)

        # Animation loop
        for tidx in range(mt-1):
            # Take one step with the fourth order Runge-Kutta method.
            w, t = rk4.step(rhs, w, t, ht)
            u, phi_t = np.split(w, 2)
            
            if tidx % 5 == 0 and show_animation:
                update_plot(tidx)

        # Show the plot
        plt.show()

    # Close figure window
    if show_animation:
        plt.close()

    return u, T, xvec, yvec, hx, L, c, w, k, w_ang


def l2_norm(vec, h):
    return np.sqrt(h) * np.sqrt(np.sum(vec**2))


def compute_error(u, u_exact, h):
    error_vec = u - u_exact
    l2_error = l2_norm(error_vec, h)
    return l2_error


def plot_final_solution(u, u_exact, xvec, T):
    fig, ax = plt.subplots()
    ax.plot(xvec, u, label='Approximation')
    plt.plot(xvec, u_exact, 'r--', label='Exact')
    ax.set_xlim([xvec[0], xvec[-1]])
    ax.set_ylim([-1, 1.2])
    plt.title(f't = {T:.2f}')
    plt.legend()
    plt.show()


def main():
    ms = [250, 50, 100, 200, 400]
    orders = [6]

    # Run all simulations
    err_dict = {}
    h_space = {}
    for o in orders:
        err_dict[o] = []
        h_space[o] = []
        for m in ms:
            u, T, xvec, yvec, h, L, c, w, k, w_ang = run_simulation(200, 100, o, True) # mx=200, my=100
            #u_exact = exact_solution(T, xvec, yvec, w_ang, k)
            #err = compute_error(u, u_exact, h)
            err_dict[o].append(err)
            h_space[o].append(h)

    # Compute convergence rates
    q_dict = {}
    for o in orders:
        q_dict[o] = [0.0]  # Set first rate to 0
        err_vec = err_dict[o]

        for i in range(len(err_vec) - 1):
            q_dict[o].append(np.log(err_vec[i] / err_vec[i + 1]) / np.log(ms[i + 1] / ms[i]))

    # Print
    for o in orders:
        print(f'----- Order: {o} ------')
        for m, err, q in zip(ms, err_dict[o], q_dict[o]):
            print(f'{m}\t{err:.2e}\t{q:.2f}')

        plt.loglog(h_space[o], err_dict[o], 'o', label=f'Data Points')

        # Add labels and legend
        plt.xlabel('Grid Spacing (h)')
        plt.ylabel('L2-Error')
        plt.legend()
        plt.title(f'Convergence Plot For Order {o}')

        # Show the plot
        plt.show()


def exact_solution(T, xvec, yvec, w_ang, k):
    u_exact = sol(xvec, yvec, T, w_ang, k)
    return u_exact


def sol(x, y, t, w_ang, k):
    return np.cos(np.pi * x) * np.cos(np.pi * y) * np.cos(w_ang * t)


def initial(x, y):
    return np.exp(-(x**2 + y**2) / 0.025)


if __name__ == '__main__':
    main()
