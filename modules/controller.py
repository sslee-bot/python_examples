import numpy as np


# Wrap angle in radians to [-pi, pi]
def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def activation_sigmoid(values):
    output = np.zeros_like(values)
    for i in range(len(values)):
        output[i] = 1.0 / (1.0 + np.exp(-values[i]))
    return output


class Kinematic:
    def __init__(self, gamma1, gamma2, h):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.h = h
        self.v = np.zeros(2)

    def control(self, x, x_ref):
        delta = x_ref - x
        delta[2] = wrap_angle(delta[2])
        theta_r = wrap_angle(x_ref[2])

        e = np.linalg.norm(delta[0:2])
        phi = theta_r - np.angle(delta[0] + 1j * delta[1])
        phi = wrap_angle(phi)
        alpha = phi - delta[2]
        alpha = wrap_angle(alpha)

        self.v[0] = self.gamma1 * e * np.cos(alpha)
        if alpha == 0:
            self.v[1] = 0.0
        else:
            self.v[1] = -self.gamma2 * alpha - self.gamma1 * np.cos(alpha) * np.sin(alpha) / alpha * (
                        alpha + self.h * phi)
        return self.v


class NN_Lewis:
    def __init__(self, dt, gamma1, gamma2, h, k4=1.0, kappa=0.1, coeff_F=0.01, coeff_G=0.01, num_neuron=10):
        self.dt = dt
        self.kinematic = Kinematic(gamma1, gamma2, h)
        self.vc = np.zeros(2)
        self.vc_pre = np.zeros(2)
        self.vc_dot = np.zeros(2)
        num_nn_input = 7  # elements: '1', vc, vc_dot, v
        self.num_nn_input = num_nn_input
        self.num_neuron = num_neuron
        self.k4 = k4
        self.kappa = kappa
        self.V = np.zeros((num_nn_input, num_neuron))
        self.W = np.zeros((num_neuron, 2))
        self.F = coeff_F * np.eye(num_neuron)
        self.G = coeff_G * np.eye(num_nn_input)
        self.f_hat = np.zeros(2)
        self.tau_bar = np.zeros(2)

    def neural_network(self, nn_input, ec):
        # Tuning NN weights
        sigma = activation_sigmoid(self.V.transpose() @ nn_input)
        sigma_prime = np.diag(sigma) @ (np.eye(self.num_neuron) - np.diag(sigma))
        W_dot = np.outer(self.F @ sigma, ec) - np.outer(self.F @ sigma_prime @ self.V.transpose() @ nn_input,
                                                        ec) - self.kappa * np.linalg.norm(ec) * self.F @ self.W
        V_dot = np.outer(self.G @ nn_input, (sigma_prime.transpose() @ self.W @ ec)) - self.kappa * np.linalg.norm(
            ec) * self.G @ self.V
        self.W += W_dot * self.dt
        self.V += V_dot * self.dt
        # NN output
        self.f_hat = self.W.transpose() @ sigma

    def control(self, x, x_ref, v):
        ec = self.vc - v
        nn_input = np.array([1.0, *self.vc, *self.vc_dot, *v])
        self.neural_network(nn_input, ec)
        self.tau_bar = self.f_hat + self.k4 * ec

        self.vc_pre = self.vc.copy()
        self.vc = self.kinematic.control(x, x_ref)
        self.vc_dot = (self.vc - self.vc_pre) / self.dt
        return self.tau_bar
