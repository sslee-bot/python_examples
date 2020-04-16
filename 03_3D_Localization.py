import numpy as np
import sympy as sym
from scipy.io import loadmat
from modules import estimator
import matplotlib.pyplot as plt

# State variables, Control inputs, Time-varying parameters
x, y, z, x_dot, y_dot, z_dot = sym.symbols('x y z x_dot y_dot z_dot')
dt = sym.symbols('dt')
state_set = (x, y, z, x_dot, y_dot, z_dot)
input_set = (dt,)
# Position values of fixed anchors #1 ~ #4
anchor_pos = np.array([[10.8, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 6.5, 0.0], [10.8, 6.5, 2.0]])
# State and measurement equations
f = sym.Matrix(1, 6, [0, 0, 0, 0, 0, 0])
f[0] = x + x_dot * dt
f[1] = y + y_dot * dt
f[2] = z + z_dot * dt
f[3] = x_dot
f[4] = y_dot
f[5] = z_dot
h = sym.Matrix(1, 4, [0, 0, 0, 0])
h[0] = sym.sqrt((x - anchor_pos[0][0]) ** 2 + (y - anchor_pos[0][1]) ** 2 + (z - anchor_pos[0][2]) ** 2)
h[1] = sym.sqrt((x - anchor_pos[1][0]) ** 2 + (y - anchor_pos[1][1]) ** 2 + (z - anchor_pos[1][2]) ** 2)
h[2] = sym.sqrt((x - anchor_pos[2][0]) ** 2 + (y - anchor_pos[2][1]) ** 2 + (z - anchor_pos[2][2]) ** 2)
h[3] = sym.sqrt((x - anchor_pos[3][0]) ** 2 + (y - anchor_pos[3][1]) ** 2 + (z - anchor_pos[3][2]) ** 2)

# Load measurement, control input data
RTLS_data = loadmat('real_data/03_Exp_data.mat')
msr_data = RTLS_data['measurement_data']
sampling_time = RTLS_data['sampling_time']

# Initialize estimator
x_init = np.array([5.5, 3.5, 1.0, 0.0, 0.0, 0.0])
z_init = msr_data[0]
u_init = sampling_time[0]
P = np.zeros((6, 6))
Q = np.diag([0.3, 0.3, 0.3, 1.0, 1.0, 1.0])
R = np.diag([0.2, 0.2, 0.2, 0.2])
Est1 = estimator.EKF(f, h, state_set, input_set, P, Q, R, x_init)
Est2 = estimator.PF_Gaussian(f, h, state_set, input_set, 1000, P, Q, R, x_init)
Est3 = estimator.FIR(f, h, state_set, input_set, 20, x_init, z_init, u_init)

# Simulation
num_iteration = msr_data.shape[0]
EKF_data = np.zeros((num_iteration, len(state_set)))
PF_data = np.zeros((num_iteration, len(state_set)))
FIR_data = np.zeros((num_iteration, len(state_set)))
for i in range(num_iteration):
    print(i)
    msr = msr_data[i]
    ctrl = sampling_time[i]
    alpha = not (0 in msr)

    x_hat_EKF = Est1.estimate(msr, ctrl, alpha)
    x_hat_PF = Est2.estimate(msr, ctrl, alpha)
    x_hat_FIR = Est3.estimate(msr, ctrl, alpha)
    EKF_data[i] = x_hat_EKF
    PF_data[i] = x_hat_PF
    FIR_data[i] = x_hat_FIR

# Plot Results
plt.figure(figsize=(15, 5))
plt.subplot(131, projection='3d')
plt.plot(EKF_data[:, 0], EKF_data[:, 1], EKF_data[:, 2])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('EKF')

plt.subplot(132, projection='3d')
plt.plot(PF_data[:, 0], PF_data[:, 1], PF_data[:, 2])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('PF')

plt.subplot(133, projection='3d')
plt.plot(FIR_data[:, 0], FIR_data[:, 1], FIR_data[:, 2])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('FIR')
plt.show()
