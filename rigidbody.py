import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class RigidBody:
    def __init__(
        self,
        mass,
        inertia_tensor,
        initial_position,
        initial_orientation,
        initial_velocity=None,
    ):
        self.mass = mass
        self.inertia_tensor = inertia_tensor
        self.inv_inertia_tensor = np.linalg.inv(inertia_tensor)
        self.position = np.array(initial_position, dtype=float)
        self.orientation = Rotation.from_quat(initial_orientation)
        if initial_velocity is None:
            self.linear_velocity = np.zeros(3)
        else:
            self.linear_velocity = initial_velocity
        self.angular_velocity = np.zeros(3)
        self.force = np.zeros(3)  # in body frame
        self.torque = np.zeros(3)  # in body frame
        self.position_history = [self.position.copy()]
        self.orientation_history = [self.orientation]

    # both force and point are in the same representation, either body frame or world frame
    def apply_force(self, force, point, world_frame):
        if world_frame:
            self.force += force
            self.torque += np.cross(point - self.position, force)
        else:
            world_force = self.orientation.apply(force)
            self.force += world_force
            self.torque += np.cross(point, force)

    def calculate_derivatives(self, state):
        position = state[:3]
        orientation_vector = state[3:6]
        linear_velocity = state[6:9]
        angular_velocity = state[9:]

        # Linear acceleration
        linear_acceleration = self.force / self.mass

        # Angular acceleration
        angular_acceleration = self.inv_inertia_tensor @ (
            self.torque
            - np.cross(angular_velocity, self.inertia_tensor @ angular_velocity)
        )

        # Orientation change rate
        angular_velocity_norm = np.linalg.norm(angular_velocity)
        if angular_velocity_norm > 1e-10:  # Avoid division by zero
            axis = angular_velocity / angular_velocity_norm
            angle = angular_velocity_norm
            orientation_dot = 0.5 * angle * np.concatenate([axis, [0]])
        else:
            orientation_dot = np.zeros(4)

        return np.concatenate(
            [
                linear_velocity,
                orientation_dot[:3],  # We only need the vector part
                linear_acceleration,
                angular_acceleration,
            ]
        )

    def update(self, dt):
        initial_state = np.concatenate(
            [
                self.position,
                self.orientation.as_rotvec(),  # Use rotation vector instead of quaternion
                self.linear_velocity,
                self.angular_velocity,
            ]
        )

        # RK4 integration
        k1 = self.calculate_derivatives(initial_state)
        k2 = self.calculate_derivatives(initial_state + 0.5 * dt * k1)
        k3 = self.calculate_derivatives(initial_state + 0.5 * dt * k2)
        k4 = self.calculate_derivatives(initial_state + dt * k3)

        # Update state
        state_change = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        new_state = initial_state + state_change

        # Update position and linear velocity
        self.position = new_state[:3]
        self.linear_velocity = new_state[6:9]

        # Update orientation
        new_orientation_vector = new_state[3:6]
        self.orientation = self.orientation * Rotation.from_rotvec(
            new_orientation_vector
        )

        # Update angular velocity
        self.angular_velocity = new_state[9:]

        # Reset forces and torques
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

        # Store position for plotting
        self.position_history.append(self.position.copy())
        self.orientation_history.append(self.orientation)

    def get_state(self):
        return {
            "position": self.position,
            "orientation": self.orientation.as_quat(),
            "linear_velocity": self.linear_velocity,
            "angular_velocity": self.angular_velocity,
        }

    def plot_orientation(self, ax, r, offset, scale=1):
        colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
        loc = np.array([offset, offset])
        for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis), colors)):
            axlabel = axis.axis_name
            axis.set_label_text(axlabel)
            axis.label.set_color(c)
            axis.line.set_color(c)
            axis.set_tick_params(colors=c)
            line = np.zeros((2, 3))
            line[1, i] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
            text_loc = line[1] * 1.2
            text_loc_rot = r.apply(text_loc)
            text_plot = text_loc_rot + loc[0]
            ax.text(*text_plot, axlabel.upper(), color=c, va="center", ha="center")
        # ax.text(*offset, name, color="k", va="center", ha="center",
        #         bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})

    def plot_trajectory(self, every_n):
        positions = np.array(self.position_history)
        orientations = self.orientation_history

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        index_list = range(0, len(positions))
        index_list = index_list[0::every_n]

        for i in index_list:
            ax.plot(positions[i, 0], positions[i, 1], positions[i, 2], "b.")
            self.plot_orientation(ax, orientations[i], positions[i, :], 1)

        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            c="g",
            label="Start",
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            c="r",
            label="End",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Rigid Body Trajectory")
        ax.legend()
        ax.axis("equal")

        plt.show()
