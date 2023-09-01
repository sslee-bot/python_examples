import numpy as np
import sympy as sym
from modules import controller
import matplotlib.pyplot as plt


def update_state(pose, control_input, base_to_center, sampling_time):
    # pose: x, y, theta
    # control_input: linear velocity, angular velocity
    S = np.array(
        [[np.cos(pose[2]), -base_to_center * np.sin(pose[2])], [np.sin(pose[2]), base_to_center * np.cos(pose[2])],
         [0, 1]])
    pose_dot = S @ control_input
    output = pose + pose_dot * sampling_time
    output[2] = controller.wrap_angle(output[2])
    return output


# Mass, inertia, wheelbase to center, wheel radius, and wheel to center line
m = 1.0
I = 1.0
d = 0.1
r = 0.05
R = 0.2

dt = 0.1

theta, theta_dot = sym.symbols('theta theta_dot')
tau_right, tau_left = sym.symbols('tau_right tau_left')
tau = sym.Matrix([tau_right, tau_left])
t = sym.symbols('t')

v_lin, v_ang = sym.symbols('v_lin v_ang')
v_lin_dot, v_ang_dot = sym.symbols('v_lin_dot v_ang_dot')
v = sym.Matrix([v_lin, v_ang])
v_dot = sym.Matrix([v_lin_dot, v_ang_dot])

# Matrices for Euler-Lagrange equation
M = sym.Matrix([[m, 0.0, m * d * sym.sin(theta)], [0.0, m, -m * d * sym.cos(theta)],
                [m * d * sym.sin(theta), -m * d * sym.sin(theta), I]])
V = sym.Matrix(
    [[0.0, 0.0, m * d * theta_dot * sym.cos(theta)], [0.0, 0.0, m * d * theta_dot * sym.sin(theta)], [0.0, 0.0, 0.0]])
F = sym.zeros(3, 1)
tau_d = sym.zeros(3, 1)
B = 1.0 / r * sym.Matrix([[sym.cos(theta), sym.cos(theta)], [sym.sin(theta), sym.sin(theta)], [R, -R]])

S = sym.Matrix([[sym.cos(theta), -d * sym.sin(theta)], [sym.sin(theta), d * sym.cos(theta)], [0.0, 1.0]])
S_dot = sym.Matrix([[-sym.sin(theta) * theta_dot, -d * sym.cos(theta) * theta_dot],
                    [sym.cos(theta) * theta_dot, -d * sym.sin(theta) * theta_dot], [0.0, 0.0]])

M_bar = sym.transpose(S) * M * S
V_bar = sym.transpose(S) * (M * S_dot + V * S)
F_bar = sym.transpose(S) * F
tau_d_bar = sym.transpose(S) * tau_d
B_bar = sym.transpose(S) * B

# Define and solve the equation
eq = M_bar * v_dot + V_bar * v + F_bar + tau_d_bar - B_bar * tau
eq1 = sym.Eq(eq[0], 0)
eq2 = sym.Eq(eq[1], 0)
v_dot_sol = sym.solve([eq1, eq2], v_dot)

# Initialization for Simulation
q = np.zeros(3)
q_ref = np.array([5.0, 5.0, 0.0])
v_present = np.zeros(2)
v_dot_present = np.zeros(2)
time_sequence = np.arange(0.0, 10.0, dt)
num_iteration = len(time_sequence)

q_data = np.zeros((num_iteration, 3))
v_data = np.zeros((num_iteration, 2))

gamma1 = 0.3
gamma2 = 0.1
h = 0.5
Ctrl1 = controller.NN_Lewis(dt, gamma1, gamma2, h)

for i in range(num_iteration):
    # Save data
    q_data[i] = q
    v_data[i] = v_present

    # Get control input
    B_bar_present = np.array(B_bar.subs(theta, q[2])).astype(float)
    tau = np.linalg.inv(B_bar_present) @ Ctrl1.control(q, q_ref, v_present)

    # Obtain v_present
    v_dot_present[0] = v_dot_sol[v_lin_dot].subs([(tau_right, tau[0]), (tau_left, tau[1])])
    v_dot_present[1] = v_dot_sol[v_ang_dot].subs(
        [(theta, q[2]), (theta_dot, v_present[1]), (v_lin, v_present[0]), (v_ang, v_present[1]), (tau_right, tau[0]),
         (tau_left, tau[1])])
    v_present += v_dot_present * dt

    # Update
    q = update_state(q, v_present, d, dt)
    print(q)

plt.plot(q_data[:, 0], q_data[:, 1], 'o')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()
