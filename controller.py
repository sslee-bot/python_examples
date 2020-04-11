import numpy as np


# Wrap angle in radians to [-pi, pi]
def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


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
        self.v[1] = -self.gamma2 * alpha - self.gamma1 * np.cos(alpha) * np.sin(alpha) / alpha * (alpha + self.h * phi)
        return self.v
