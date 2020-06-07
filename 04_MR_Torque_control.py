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


# Mass, wheelbase to center, wheel radius, and wheel to center line
m = 1.0
d = 0.1
r = 0.05
R = 0.2

dt = 0.5

theta, theta_dot = sym.symbols('theta theta_dot')
tau_right, tau_left = sym.symbols('tau_right tau_left')
tau = sym.Matrix([tau_right, tau_left])
t = sym.symbols('t')

v_lin, v_ang = sym.symbols('v_lin v_ang', cls=sym.Function)
v_lin_dot = sym.Derivative(v_lin(t), t)
v_ang_dot = sym.Derivative(v_ang(t), t)
v = sym.Matrix([v_lin(t), v_ang(t)])
v_dot = sym.Matrix([v_lin_dot, v_ang_dot])

# Matrices for Euler-Lagrange equation
M = sym.Matrix([[m, 0.0, m * d * sym.sin(theta)], [0.0, m, -m * d * sym.cos(theta)],
                [m * d * sym.sin(theta), -m * d * sym.sin(theta), 1.0]])
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

# Define the equation
eq = M_bar * v_dot + V_bar * v + F_bar + tau_d_bar - B_bar * tau
eq1 = sym.Eq(eq[0], 0)
eq2 = sym.Eq(eq[1], 0)

v_lin_sol = sym.dsolve(eq1, v_lin(t))

# Initialization for Simulation
q = np.zeros(3)
v_present = np.zeros(2)
time_sequence = np.arange(0.0, 10.0, dt)
num_iteration = len(time_sequence)
torque_right_data = 0.03 * np.sin(time_sequence)
torque_left_data = 0.03 * np.cos(time_sequence)
control_data = np.transpose(np.vstack((torque_right_data, torque_left_data)))
q_data = np.zeros((num_iteration, 3))
v_data = np.zeros((num_iteration, 2))

for i in range(num_iteration):
    # Save data
    q_data[i] = q
    v_data[i] = v_present

    # Get control input
    tau = control_data[i]

    # Solve dynamic equation
    constants1 = sym.solve(v_lin_sol.rhs.subs(t, 0) - v_present[0], dict=True)
    v_present[0] = v_lin_sol.subs(*constants1).subs([(t, dt), (tau_right, tau[0]), (tau_left, tau[1])]).rhs

    eq_ang = eq2.subs(
        [(theta, q[2]), (theta_dot, v_present[1]), (tau_right, tau[0]), (tau_left, tau[1]), (v_lin(t), v_present[0])])
    # Here, still bug exists when i=10
    v_ang_sol = sym.dsolve(sym.simplify(eq_ang), v_ang(t))
    constants2 = sym.solve(v_ang_sol.rhs.subs(t, 0) - v_present[1], dict=True)
    v_present[1] = v_ang_sol.subs(*constants2).subs(t, dt).rhs

    # Update
    q = update_state(q, v_present, d, dt)
    print(q)

plt.plot(q_data[:, 0], q_data[:, 1], 'o')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()
