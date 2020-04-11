import numpy as np
import controller
import matplotlib.pyplot as plt


def update_state(pose, control_input, wheelbase, sampling_time):
    # pose: x, y, theta
    # control_input: linear velocity, angular velocity
    S = np.array(
        [[np.cos(pose[2]), -wheelbase * np.sin(pose[2])], [np.sin(pose[2]), wheelbase * np.cos(pose[2])], [0, 1]])
    pose_dot = S @ control_input
    output = pose + pose_dot * sampling_time
    output[2] = (output[2] + np.pi) % (2 * np.pi) - np.pi
    return output


# Initialization
x = np.zeros(3)
dt = 0.1
d = 0.1
gamma1 = 1.5
gamma2 = 0.7
h = 2

Ctrl1 = controller.Kinematic(gamma1, gamma2, h)

# Generate position reference
x_ref_1 = np.array([1, 1, 0])
x_ref_2 = np.array([5, 1, 0])
x_ref_3 = np.array([3, 5, 2 / 3 * np.pi])
x_ref_4 = np.array([1, 1, 0])

x_ref_data = np.zeros((400, 3))
x_ref_data[0:100] = np.tile(x_ref_1, (100, 1))
x_ref_data[100:200] = np.tile(x_ref_2, (100, 1))
x_ref_data[200:300] = np.tile(x_ref_3, (100, 1))
x_ref_data[300:400] = np.tile(x_ref_4, (100, 1))

# Simulation
x_data = np.zeros((400, 3))
for i in range(400):
    # Control
    x_ref = x_ref_data[i]
    v = Ctrl1.control(x, x_ref)
    # Update state
    x = update_state(x, v, d, dt)
    # Save data
    x_data[i] = x
plt.plot(x_data[:, 0], x_data[:, 1], 'o')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()
