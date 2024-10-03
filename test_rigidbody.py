from rigidbody import RigidBody
import numpy as np

if __name__ == "__main__":
    rb = RigidBody(
        1,  # mass
        np.diag([100, 10000, 1000000]),  # inertia tensor
        [0, 0, 200],  # initial position
        [0, 0, 0, 1],  # intial orientation
        [10, 0, 0],  # initial velocity)
    )

    fixed_simulation_step = 0.01
    debug_display_step = 0.1
    simulation_length = 2  # in seconds

    for t in np.arange(0, simulation_length, fixed_simulation_step):
        if abs(t % debug_display_step) < 1e-6:
            state = rb.get_state()

            print(f"Time: {t:.2f}s")
            print(f"Position: {state['position']}")
            print(f"Orientation: {state['orientation']}")
            print(f"Linear Velocity: {state['linear_velocity']}")
            print(f"Angular Velocity: {state['angular_velocity']}")
            print()

        # if abs(t % control_update_step) < 1e-6:
        #     aircraft.ApplyInput(0.1, 0.5, 0.0, 0.0)

        # aircraft.ApplyGravity()
        # aircraft.ApplyThrustForce()
        # aircraft.ApplyMainWingForce()
        # aircraft.ApplyTailWingForce()
        # aircraft.rigidbody.update(fixed_simulation_step)

        # aircraft.UpdateState(fixed_simulation_step)

    rb.plot_trajectory(10)
