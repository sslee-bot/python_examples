import numpy as np
import sympy as sym


class FIR:
    def __init__(self, f, h, state_set, input_set, N):
        self.dim_x = len(f)
        self.dim_z = len(h)
        self.dim_u = len(input_set)
        self.N = N
        self.alpha = True

        self.x_hat = np.ones(self.dim_x)
        self.z = np.zeros(self.dim_z)
        self.u = np.zeros(self.dim_u)
        self.f_hat = np.zeros(self.dim_x)
        self.h_hat = np.zeros(self.dim_z)
        # Functions
        self.f_func = sym.lambdify(state_set + input_set, f)
        self.h_func = sym.lambdify(state_set + input_set, h)
        self.Jacobian_F_func = sym.lambdify(state_set + input_set, f.jacobian(state_set))
        self.Jacobian_H_func = sym.lambdify(state_set + input_set, h.jacobian(state_set))

        # self.F = self.Jacobian_F_func(3,3,0,0,0,0.1)
        # self.H = self.Jacobian_H_func(3,3,0,0,0,0.1)
        self.F = np.eye(self.dim_x)
        self.H = np.ones((self.dim_z, self.dim_x))
        self.f_hat = np.zeros(self.dim_x)
        self.h_hat = np.zeros(self.dim_z)
        self.z_tilde = np.zeros(self.dim_z)
        self.u_tilde = np.zeros(self.dim_x)
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

    def estimate(self, z, u, alpha):
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
