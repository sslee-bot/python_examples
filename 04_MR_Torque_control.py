import numpy as np
import sympy as sym
from modules import controller
import matplotlib.pyplot as plt

m = 1.0
d = 0.1
r = 0.05
R = 0.2

dt = 0.5

theta, theta_dot = sym.symbols('theta theta_dot')

# Matrices for Euler-Lagrange equation
M = sym.Matrix([[m, 0.0, m * d * sym.sin(theta)], [0.0, m, -m * d * sym.cos(theta)],
                [m * d * sym.sin(theta), -m * d * sym.sin(theta), 1.0]])
V = sym.Matrix(
    [[0.0, 0.0, m * d * theta_dot * sym.cos(theta)], [0.0, 0.0, m * d * theta_dot * sym.sin(theta)], [0.0, 0.0, 0.0]])
F = np.zeros(3)
tau_d = np.zeros(3)
B = 1.0 / r * sym.Matrix([[sym.cos(theta), sym.cos(theta)], [sym.sin(theta), sym.sin(theta)], [R, -R]])

S = sym.Matrix([[sym.cos(theta), -d * sym.sin(theta)], [sym.sin(theta), d * sym.cos(theta)], [0.0, 1.0]])
S_dot = sym.Matrix([[-sym.sin(theta) * theta_dot, -d * sym.cos(theta) * theta_dot],
                    [sym.cos(theta) * theta_dot, -d * sym.sin(theta) * theta_dot], [0.0, 0.0]])

M_bar = sym.transpose(S) @ M @ S
V_bar = sym.transpose(S) @ (M @ S_dot + V @ S)
F_bar = sym.transpose(S) @ F
tau_d_bar = sym.transpose(S) @ tau_d
B_bar = sym.transpose(S) @ B

# Initialization for Simulation
q = np.zeros(3)
v_present = np.zeros(2)
time_sequence = np.arange(0.0, 10.0, dt)
num_iteration = len(time_sequence)
torque_left_data = 0.03 * np.sin(time_sequence)
torque_right_data = 0.03 * np.cos(time_sequence)
control_data = np.transpose(np.vstack((torque_left_data, torque_right_data)))
q_data = np.zeros((num_iteration, 3))
v_data = np.zeros((num_iteration, 2))

for i in range(num_iteration):
    # Save data
    q_data[i] = q
    v_data[i] = v_present

    # Get control input
    tau = control_data[i]

    # Solve dynamic equation
    