import numpy as np
import matplotlib.pyplot as plt


class RigidBody:
    def __init__(
        self,
        mass,
        position,
        velocity,
        inertia_tensor,
        angular_velocity=None,
        orientation=None,
    ):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.inertia_tensor = np.array(
            inertia_tensor, dtype=float
        )  # 3x3 inertia tensor
        self.angular_velocity = np.array(
            angular_velocity if angular_velocity is not None else [0.0, 0.0, 0.0],
            dtype=float,
        )
        self.orientation = np.array(
            orientation if orientation is not None else [0.0, 0.0, 0.0], dtype=float
        )  # Euler angles
        self.acceleration = np.zeros(3)
        self.angular_acceleration = np.zeros(3)

        # List to track trajectory
        self.trajectory = []

    def apply_force(self, force, point):
        """Apply a force at a specified point on the rigid body in body frame."""
        if isinstance(force, (list, tuple)):
            force = np.array(force, dtype=float)
        if isinstance(point, (list, tuple)):
            point = np.array(point, dtype=float)

        # Convert force from body frame to world frame
        rotation_matrix = self.rotation_matrix(self.orientation)
        world_force = rotation_matrix.dot(force)

        # Update linear acceleration
        self.acceleration += world_force / self.mass

        # Calculate torque: τ = r × F
        r = point - self.position
        torque = np.cross(r, world_force)

        # Update angular acceleration
        self.angular_acceleration += np.linalg.inv(self.inertia_tensor).dot(torque)

    def rotation_matrix(self, euler_angles):
        """Create a rotation matrix from Euler angles."""
        roll, pitch, yaw = euler_angles
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )

        R_y = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )

        R_z = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        return R_z @ R_y @ R_x  # Combine rotations

    def rk4_step(self, delta_time):
        """Perform a single Runge-Kutta 4th order integration step."""

        # Define the state vector for position, velocity, angular velocity
        state = np.concatenate((self.position, self.velocity, self.angular_velocity))

        def derivatives(state):
            pos = state[:3]
            vel = state[3:6]
            ang_vel = state[6:9]

            # Update the linear and angular accelerations
            self.position = pos
            self.velocity = vel
            self.angular_velocity = ang_vel

            lin_acc = self.acceleration
            ang_acc = self.angular_acceleration

            return np.concatenate((vel, lin_acc, ang_vel, ang_acc))

        # Runge-Kutta coefficients
        k1 = derivatives(state)
        k2 = derivatives(state + delta_time / 2 * k1)
        k3 = derivatives(state + delta_time / 2 * k2)
        k4 = derivatives(state + delta_time * k3)

        # Update the state using the weighted average of the slopes
        state += delta_time / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Update the object's position, velocity, and angular velocity
        self.position = state[:3]
        self.velocity = state[3:6]
        self.angular_velocity = state[6:9]

        # Reset accelerations after the update
        self.acceleration = np.zeros(3)
        self.angular_acceleration = np.zeros(3)

        # Append the current position to the trajectory
        self.trajectory.append(self.position.copy())

    def plot_trajectory(self, n=1):
        """Plot the trajectory of the rigid body, plotting every nth position."""
        trajectory = np.array(self.trajectory)

        if n > 1:
            trajectory = trajectory[::n]  # Select every nth position

        plt.figure(figsize=(10, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker="o")
        plt.title("Rigid Body Trajectory")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid()
        plt.axis("equal")
        plt.show()

    def get_position(self):
        """Return the current position of the rigid body."""
        return self.position

    def get_velocity(self):
        """Return the current velocity of the rigid body."""
        return self.velocity

    def get_angular_velocity(self):
        """Return the current angular velocity of the rigid body."""
        return self.angular_velocity

    def reset(self):
        """Reset position, velocity, and acceleration to zero."""
        self.position = np.array([0.0, 0.0, 0.0], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.acceleration = np.zeros(3)
        self.angular_acceleration = np.zeros(3)
        self.trajectory = []  # Clear trajectory

    def __str__(self):
        return (
            f"RigidBody(mass={self.mass}, position={self.position}, velocity={self.velocity}, "
            f"angular_velocity={self.angular_velocity}, orientation={self.orientation}, "
            f"inertia_tensor={self.inertia_tensor})"
        )


# Example usage
if __name__ == "__main__":
    inertia_tensor = np.diag([1.0, 1.0, 1.0])  # Example inertia tensor
    body = RigidBody(
        mass=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        inertia_tensor=inertia_tensor,
    )

    # Apply a force of (1, 0, 0) at the point (1, 0, 0) in body coordinates
    body.apply_force([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    # Simulate for a number of steps
    time_steps = 100
    delta_time = 0.1
    for _ in range(time_steps):
        body.rk4_step(delta_time)

    # Plot the trajectory, selecting every 5th position
    body.plot_trajectory(n=5)
