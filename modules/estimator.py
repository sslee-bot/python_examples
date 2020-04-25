import numpy as np
import sympy as sym


class EKF:
    def __init__(self, f, h, state_set, input_set, P=None, Q=None, R=None, x_init=None):
        # Scalar values
        self.dim_x = len(f)
        self.dim_z = len(h)
        self.dim_u = len(input_set)
        self.alpha = True
        # Initial state, measurement, control input
        if x_init is None:
            self.x_hat = np.zeros(self.dim_x)
        else:
            self.x_hat = x_init
        self.z = np.zeros(self.dim_z)
        self.u = np.zeros(self.dim_u)
        # Covariances
        if P is None:
            self.P = 0.1 * np.eye(self.dim_x)
        else:
            self.P = P
        if Q is None:
            self.Q = 0.1 * np.eye(self.dim_x)
        else:
            self.Q = Q
        if R is None:
            self.R = 0.1 * np.eye(self.dim_z)
        else:
            self.R = R
        # Functions
        self.f_func = sym.lambdify(state_set + input_set, f)
        self.h_func = sym.lambdify(state_set + input_set, h)
        self.Jacobian_F_func = sym.lambdify(state_set + input_set, f.jacobian(state_set))
        self.Jacobian_H_func = sym.lambdify(state_set + input_set, h.jacobian(state_set))

        self.f_hat = np.zeros(self.dim_x)
        self.h_hat = np.zeros(self.dim_z)
        self.F = np.eye(self.dim_x)
        self.H = np.ones((self.dim_z, self.dim_x))

        self.evaluate_current_values(self.z, self.u, self.alpha)

    def evaluate_current_values(self, z, u, alpha):
        self.z = z
        self.u = u

        args = np.concatenate((self.x_hat, self.u))
        self.F = self.Jacobian_F_func(*args)
        self.H = self.Jacobian_H_func(*args)
        self.f_hat = self.f_func(*args)[0]
        self.h_hat = self.h_func(*args)[0]
        if not alpha:
            self.z = self.h_hat

    def estimate(self, z, u, alpha=True):
        self.evaluate_current_values(z, u, alpha)

        F_tran = np.transpose(self.F)
        H_tran = np.transpose(self.H)
        self.P = self.F @ self.P @ F_tran + self.Q
        S = self.H @ self.P @ H_tran + self.R
        K = self.P @ H_tran @ np.linalg.inv(S)
        self.x_hat = self.f_hat + K @ (self.z - self.h_hat)
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        return self.x_hat


class PF_Gaussian:
    def __init__(self, f, h, state_set, input_set, num_particle=500, P=None, Q=None, R=None, x_init=None):
        # Scalar values
        self.dim_x = len(f)
        self.dim_z = len(h)
        self.dim_u = len(input_set)
        self.alpha = True
        # Initial state, measurement, control input
        if x_init is None:
            self.x_hat = np.zeros(self.dim_x)
        else:
            self.x_hat = x_init
        self.z = np.zeros(self.dim_z)
        self.u = np.zeros(self.dim_u)
        # Covariances
        if P is None:
            self.P = 0.1 * np.eye(self.dim_x)
        else:
            self.P = P
        if Q is None:
            self.Q = 0.1 * np.eye(self.dim_x)
        else:
            self.Q = Q
        if R is None:
            self.R = 0.1 * np.eye(self.dim_z)
        else:
            self.R = R
        # Functions
        self.f_func = sym.lambdify(state_set + input_set, f)
        self.h_func = sym.lambdify(state_set + input_set, h)
        # Particles
        self.num_particle = num_particle
        self.particle = np.random.multivariate_normal(self.x_hat, self.P, self.num_particle)
        self.weight = np.ones(self.num_particle) / self.num_particle
        self.z_hat_particle = np.zeros((self.num_particle, self.dim_z))

    def evaluate_current_values(self, z, u, alpha):
        self.z = z
        self.u = u
        self.alpha = alpha
        if not alpha:
            args = np.concatenate((self.x_hat, self.u))
            self.z = self.h_func(*args)[0]

    def propagate(self):
        for i in range(self.num_particle):
            args = np.concatenate((self.particle[i], self.u))
            self.particle[i] = self.f_func(*args)[0]
        noise_propagate = np.random.multivariate_normal(np.zeros(self.dim_x), self.Q, self.num_particle)
        self.particle += noise_propagate

    def predict_output(self):
        for i in range(self.num_particle):
            args = np.concatenate((self.particle[i], self.u))
            self.z_hat_particle[i] = self.h_func(*args)[0]

    def evaluate_weight(self):
        for i in range(self.num_particle):
            error = self.z - self.z_hat_particle[i]
            self.weight[i] = self.weight[i] * 1.0 / (
                    (2.0 * np.pi) ** (self.dim_z / 2.0) * np.linalg.det(self.R) ** 0.5) * np.exp(
                -np.transpose(error) @ np.linalg.inv(self.R) @ error / 2.0)
        sum_weight = np.sum(self.weight)
        self.weight /= sum_weight

    def resampling(self):
        num_eff = 1.0 / np.sum(self.weight ** 2)
        if num_eff < 0.1 * self.num_particle:
            temp_particle = np.copy(self.particle)
            q_cumsum = np.cumsum(self.weight)
            rand_vector = np.random.rand(self.num_particle) * q_cumsum[-1]
            for i in range(self.num_particle):
                for j in range(self.num_particle):
                    if rand_vector[i] < q_cumsum[j]:
                        self.particle[i] = temp_particle[j]
                        break
            self.weight = np.ones(self.num_particle) / self.num_particle

    def estimate(self, z, u, alpha=True):
        self.evaluate_current_values(z, u, alpha)
        self.propagate()
        self.predict_output()
        self.evaluate_weight()
        self.resampling()
        self.x_hat = self.weight @ self.particle
        return self.x_hat


class FIR:
    def __init__(self, f, h, state_set, input_set, N, x_init=None, z_init=None, u_init=None):
        # Scalar values
        self.dim_x = len(f)
        self.dim_z = len(h)
        self.dim_u = len(input_set)
        self.N = N
        self.alpha = True
        # Initial state, measurement, control input
        if x_init is None:
            self.x_hat = np.zeros(self.dim_x)
        else:
            self.x_hat = x_init
        if z_init is None:
            self.z = np.zeros(self.dim_z)
        else:
            self.z = z_init
        if u_init is None:
            self.u = np.zeros(self.dim_u)
        else:
            self.u = u_init
        # Functions
        self.f_func = sym.lambdify(state_set + input_set, f)
        self.h_func = sym.lambdify(state_set + input_set, h)
        self.Jacobian_F_func = sym.lambdify(state_set + input_set, f.jacobian(state_set))
        self.Jacobian_H_func = sym.lambdify(state_set + input_set, h.jacobian(state_set))

        self.f_hat = np.zeros(self.dim_x)
        self.h_hat = np.zeros(self.dim_z)
        self.F = np.eye(self.dim_x)
        self.H = np.ones((self.dim_z, self.dim_x))
        self.f_hat = np.zeros(self.dim_x)
        self.h_hat = np.zeros(self.dim_z)
        self.z_tilde = np.zeros(self.dim_z)
        self.u_tilde = np.zeros(self.dim_x)

        self.evaluate_current_values(self.z, self.u, self.alpha)
        self.evaluate_tilde()

        self.F_array = np.tile(self.F, (N, 1, 1))
        self.H_array = np.tile(self.H, (N, 1, 1))
        self.z_tilde_array = np.tile(self.z_tilde, (N, 1, 1))
        self.u_tilde_array = np.tile(self.u_tilde, (N, 1, 1))

    def evaluate_current_values(self, z, u, alpha):
        self.z = z
        self.u = u
        self.alpha = alpha
        args = np.concatenate((self.x_hat, self.u))
        self.F = self.Jacobian_F_func(*args)
        self.H = self.Jacobian_H_func(*args)
        self.f_hat = self.f_func(*args)[0]
        self.h_hat = self.h_func(*args)[0]
        if not alpha:
            self.z = self.h_hat

    def evaluate_tilde(self):
        self.z_tilde = self.z - (self.h_hat - self.H @ self.f_hat)
        self.u_tilde = self.f_hat - self.F @ self.x_hat

    def update_array(self):
        self.F_array = np.concatenate((self.F_array[1:], [self.F]))
        self.H_array = np.concatenate((self.H_array[1:], [self.H]))
        self.z_tilde_array = np.concatenate((self.z_tilde_array[1:], [[self.z_tilde]]))
        self.u_tilde_array = np.concatenate((self.u_tilde_array[1:], [[self.u_tilde]]))

    def Mat_F(self, a, c):  # F_function
        output = np.eye(self.dim_x)
        if a < self.N:
            for i in range(self.N - a):
                output = output @ self.F_array[-i - c]
        return output

    def Mat_Gamma(self, a, b, c):  # Gamma_function
        if a < b:
            return np.zeros((self.dim_z, self.dim_x))
        else:
            output = self.H_array[-c - self.N + a]
            for i in range(a - b):
                output = output @ self.F_array[-i - 1 - c - self.N + a]
            return output

    def A_big(self):
        output = np.zeros((self.N * self.dim_z, self.dim_x))
        for i in range(self.N):
            output[self.dim_z * i:self.dim_z * (i + 1)] = self.Mat_Gamma(i + 1, 1, 1)
        return output

    def B_big(self):
        output = np.zeros((self.N * self.dim_z, self.N * self.dim_x))
        for i in range(self.N - 1):
            for j in range(self.N - 1):
                row_start = self.dim_z * (i + 1)
                row_end = self.dim_z * (i + 2)
                col_start = self.dim_x * j
                col_end = self.dim_x * (j + 1)
                output[row_start:row_end, col_start:col_end] = self.Mat_Gamma(i + 1, j + 1, 0)
        return output

    def C_big(self):
        output = np.zeros((self.dim_x, self.N * self.dim_x))
        for i in range(self.N):
            col_start = self.dim_x * i
            col_end = self.dim_x * (i + 1)
            output[:, col_start:col_end] = self.Mat_F(i + 1, 1)
        return output

    def estimate(self, z, u, alpha=True):
        self.evaluate_current_values(z, u, alpha)
        self.evaluate_tilde()
        self.update_array()
        A = self.A_big()
        A_tran = np.transpose(A)
        B = self.B_big()
        C = self.C_big()
        F_0 = self.Mat_F(0, 1)
        L = F_0 @ np.linalg.inv(A_tran @ A) @ A_tran
        M = -L @ B + C
        z_tilde_vector = np.reshape(self.z_tilde_array, (1, -1))[0]
        u_tilde_vector = np.reshape(self.u_tilde_array, (1, -1))[0]
        self.x_hat = L @ z_tilde_vector + M @ u_tilde_vector
        return self.x_hat
