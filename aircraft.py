import math
import pymunk
import pymunk.pygame_util
import pygame
import pymunk.vec2d

from lift_coef_data import *


class Aircraft:
    def __init__(self, height, x_dist_to_landing):
        self.window_width = 3500
        self.window_height = 820
        self.window = pygame.display.set_mode((self.window_width, self.window_height))

        self.draw_options = pymunk.pygame_util.DrawOptions(self.window)

        self.y_offset = 800  # ground is at 800. So if height is h, then its actual height is 800 - h
        self.x_landing_point = x_dist_to_landing

        self.Reset(height)

        self.propeller_rpm_min = 675
        self.propeller_rpm_max = 3000
        self.propeller_rpm_rate = 500

        self.elevator_max_angle = 15.0

        self.tail_wing_trim_angle_change_rate = 1.0
        self.tail_wing_trim_angle_max = 10.0
        self.tail_wing_trim_angle_min = -10.0

    def Reset(self, height):
        self.space = pymunk.Space()
        self.space.gravity = (0, 9.81)
        self.ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.ground_body.position = (2906.0, self.window_height - 10)
        self.ground_shape = pymunk.Poly.create_box(self.ground_body, (800, 20))
        self.ground_shape.color = (255, 255, 255, 100)
        self.space.add(self.ground_body, self.ground_shape)

        self.rigidbody = pymunk.Body(600, 1825)
        self.rigidbody.position = (0, self.y_offset - height)
        self.rigidbody.center_of_gravity = (1.5, 0.35)
        self.rigidbody.velocity = (31, 0)
        self.shape = pymunk.Poly.create_box(self.rigidbody, (12, 4))
        self.shape.color = (255, 255, 0, 100)
        self.space.add(self.rigidbody, self.shape)

        self.rigidbody.position = (0, self.y_offset - height)
        self.rigidbody.center_of_gravity = (1.5, 0.35)
        self.rigidbody.velocity = (31, 0)

        self.space.reindex_shape(self.shape)
        self.space.reindex_shapes_for_body(self.rigidbody)

        self.propeller_rpm = 1500
        self.propeller_rpm_target = 1500
        self.elevator_angle = 0
        self.tail_wing_trim_angle = 0
        self.tail_wing_trim_angle_target = 0
        self.angle_of_attack = (
            -4.0
        )  # this is fuselage AOA, initialized to main_wing_mounting_angle
        self.trajectory = []

        self.air_density = ((1.0 - height / 0.3048 / 145454.0) ** 4.2493) * 1.225
        self.path_angle = 0

        self.steps = 0
        self.path_degree = math.degrees(math.atan(height / self.x_landing_point))

    def Step(self, action, dt):
        self.steps += 1

        self.UpdateState(dt)
        thrust = action[0] + 0.63
        self.ApplyInput(thrust, 0.0, -0.33)
        self.ApplyThrustForce()
        self.ApplyMainWingForce()
        self.ApplyTailWingForce()
        self.space.step(dt)

        # add return values
        new_state = self.GetState()
        reward = 0
        done = False
        info = ""

        if self.steps > 10000:
            done = True
            info = "step>10000"
            return new_state, reward, done, info

        if self.rigidbody.position.y > 798:
            done = True
            if (
                self.rigidbody.position.x >= self.x_landing_point - 10
                and self.rigidbody.position.x <= self.x_landing_point + 10
            ):
                reward = 100
                info = "succeeded perfect"
            elif (
                self.rigidbody.position.x >= self.x_landing_point - 50
                and self.rigidbody.position.x <= self.x_landing_point + 50
            ):
                reward = 75
                info = "succeeded normal"
            elif (
                self.rigidbody.position.x > self.x_landing_point + 50
                and self.rigidbody.position.x <= self.x_landing_point + 150
            ):
                reward = 50
                info = "succeeded normal"
            else:
                reward = -100
                info = "fail"
            return new_state, reward, done, info

        if new_state[3] > 0 and new_state[3] < 5:  # body_aoa
            reward += 1

        if new_state[4] > -400 and new_state[4] < 0:
            reward += 1

        if new_state[5] > 2.8 and new_state[5] < 3.2:
            reward += 3

        return new_state, reward, done, info

    def ApplyInput(
        self,
        throttle,  # from 0-1
        elevator,  # from -1 to 1
        trim,  # from -1 to 1
    ):
        self.propeller_rpm_target = (
            throttle * (self.propeller_rpm_max - self.propeller_rpm_min)
            + self.propeller_rpm_min
        )

        self.elevator_angle = self.elevator_max_angle * elevator

        self.tail_wing_trim_angle_target = self.tail_wing_trim_angle_max * trim

    def GetState(self):
        forward_direction = self.rigidbody.rotation_vector
        projected_velocity = self.rigidbody.velocity.projection(forward_direction)
        state = [
            self.y_offset - self.rigidbody.position.y,  # height
            self.x_landing_point - self.rigidbody.position.x,  # x_dist_to_landing_point
            projected_velocity.length,  # airspeed
            self.angle_of_attack,  # body_aoa
            self.rigidbody.velocity.y * 60 / 0.3048,  # vertical_rate
            math.degrees(
                math.atan(
                    (self.y_offset - self.rigidbody.position.y)
                    / (self.x_landing_point - self.rigidbody.position.x)
                )
            ),  # path_degree
        ]
        return state

    def UpdateState(self, dt):
        state = self.GetState()
        self.trajectory.append(state)

        velocity = self.rigidbody.velocity
        position = self.rigidbody.position
        x = position.x
        y = self.y_offset - position.y

        # update air density
        self.air_density = ((1.0 - y / 0.3048 / 145454.0) ** 4.2493) * 1.225

        # update angle of attack
        forward_direction = self.rigidbody.rotation_vector
        projected_velocity = velocity.projection(forward_direction)
        self.angle_of_attack = forward_direction.get_angle_degrees_between(
            velocity
        )  # a.get_angle_degrees_between(b) returns the angle from a to b. clockwise is "+". result is (-180, 180].
        # print(f"Fuselage AOA: {self.angle_of_attack}")

        # Update propeller_rpm
        if self.propeller_rpm < self.propeller_rpm_target:
            self.propeller_rpm += dt * self.propeller_rpm_rate
            if self.propeller_rpm > self.propeller_rpm_max:
                self.propeller_rpm = self.propeller_rpm_max
        else:
            self.propeller_rpm -= dt * self.propeller_rpm_rate
            if self.propeller_rpm < self.propeller_rpm_min:
                self.propeller_rpm = self.propeller_rpm_min

        # Update trim angle
        if self.tail_wing_trim_angle < self.tail_wing_trim_angle_target:
            self.tail_wing_trim_angle += dt * self.tail_wing_trim_angle_change_rate
            if self.tail_wing_trim_angle > self.tail_wing_trim_angle_max:
                self.tail_wing_trim_angle = self.tail_wing_trim_angle_max
        else:
            self.tail_wing_trim_angle -= dt * self.tail_wing_trim_angle_change_rate
            if self.tail_wing_trim_angle < self.tail_wing_trim_angle_min:
                self.tail_wing_trim_angle = self.tail_wing_trim_angle_min

    def ApplyMainWingForce(self):
        velocity = self.rigidbody.velocity
        rotation_vector = self.rigidbody.rotation_vector

        coef_due_to_flap = 1.0  # suppose flap is fully extended
        main_wing_area = 7.5
        aileron_impact_coef = 0
        main_wing_mounting_angle = -4.0

        # Calculate main wing lift
        main_wing_angle_of_attack = self.angle_of_attack + main_wing_mounting_angle
        while main_wing_angle_of_attack > 180:
            main_wing_angle_of_attack -= 180
        while main_wing_angle_of_attack < -180:
            main_wing_angle_of_attack += 180
        # main_wing mounting angle is -4.0 degrees
        # print(f"Main Wing AOA: {main_wing_angle_of_attack}")

        coef_lookup_index = math.floor(main_wing_angle_of_attack * 10.0) + 1800
        if coef_lookup_index < 0 or coef_lookup_index >= 3601:
            print(f"main_wing_angle_of_attack: {main_wing_angle_of_attack}")
        main_wing_lift_coef = (
            lift0_coef_table[coef_lookup_index]
            + (
                lift30_coef_table[coef_lookup_index]
                - lift0_coef_table[coef_lookup_index]
            )
            * coef_due_to_flap
        )
        left_wing_lift_force_magnitude = (
            self.air_density
            / 2.0
            * (velocity.length**2.0)
            * main_wing_lift_coef
            * main_wing_area
        )  # main wing area = 7.5

        left_wing_local_coordinate = pymunk.vec2d.Vec2d(1.15, -0.3)
        left_wing_multiplier_due_to_2_degree_roll_angle = math.cos(math.radians(2))

        left_wing_lift_direction = rotation_vector.rotated_degrees(
            -90
        ).normalized()  # clockwise is positive
        # print(
        #     f"Main Wing Lift direction: {left_wing_lift_direction}, force: {left_wing_lift_force_magnitude}"
        # )
        left_wing_lift_force = (
            left_wing_lift_direction
            * left_wing_lift_force_magnitude
            * left_wing_multiplier_due_to_2_degree_roll_angle
        )  # aileron impact coef = 0

        self.rigidbody.apply_force_at_local_point(
            2 * left_wing_lift_force, left_wing_local_coordinate
        )

        # Calculate main wing drag
        main_wing_drag_coef = (
            drag0_coef_table[coef_lookup_index]
            + (
                drag30_coef_table[coef_lookup_index]
                - drag0_coef_table[coef_lookup_index]
            )
            * coef_due_to_flap
        )
        left_wing_drag_force_magnitude = (
            self.air_density
            / 2.0
            * (velocity.length**2.0)
            * main_wing_drag_coef
            * main_wing_area
        )
        left_wing_drag_direction = rotation_vector.rotated_degrees(
            -180
        ).normalized()  # clockwise is positive
        # print(
        #     f"Main Wing Drag direction: {left_wing_drag_direction}, force: {left_wing_drag_force_magnitude}"
        # )
        left_wing_drag_force = left_wing_drag_direction * left_wing_drag_force_magnitude

        self.rigidbody.apply_force_at_local_point(
            2 * left_wing_drag_force, left_wing_local_coordinate
        )

    def ApplyTailWingForce(self):
        velocity = self.rigidbody.velocity
        rotation_vector = self.rigidbody.rotation_vector

        tail_wing_initial_shooting_angle = -7.4
        tail_wing_static_area = 1.5

        # Calculate tail wing down force
        tail_wing_angle_of_attack = (
            self.angle_of_attack
            + tail_wing_initial_shooting_angle
            + self.elevator_angle
            + self.tail_wing_trim_angle
        )
        while tail_wing_angle_of_attack > 180:
            tail_wing_angle_of_attack -= 180
        while tail_wing_angle_of_attack < -180:
            tail_wing_angle_of_attack += 180
        tail_static_coef_lookup_index = (
            math.floor(tail_wing_angle_of_attack * 10.0) + 1800
        )
        if tail_static_coef_lookup_index < 0 or tail_static_coef_lookup_index >= 3601:
            print(f"tail_wing_angle_of_attack: {tail_wing_angle_of_attack}")
        tail_wing_lift_coef = lift0_coef_table[tail_static_coef_lookup_index]
        left_tail_wing_lift_force_magnitude = (
            self.air_density
            / 2.0
            * (velocity.length**2.0)
            * tail_wing_lift_coef
            * tail_wing_static_area
        )
        left_tail_wing_local_coordinate = pymunk.vec2d.Vec2d(-2.9, -0.1)

        left_tail_wing_lift_direction = rotation_vector.rotated_degrees(
            -90
        ).normalized()  # clockwise is positive
        # print(
        #     f"Tail Wing Lift direction: {left_tail_wing_lift_direction}, force: {left_tail_wing_lift_force_magnitude}"
        # )
        left_tail_wing_force = (
            left_tail_wing_lift_direction * left_tail_wing_lift_force_magnitude
        )

        self.rigidbody.apply_force_at_local_point(
            2 * left_tail_wing_force, left_tail_wing_local_coordinate
        )

        # Calculate tail wing drag
        tail_wing_drag_coef = drag0_coef_table[tail_static_coef_lookup_index]
        left_tail_wing_drag_force_magnitude = (
            self.air_density
            / 2.0
            * (velocity.length**2.0)
            * tail_wing_drag_coef
            * tail_wing_static_area
        )
        left_tail_wing_drag_direction = rotation_vector.rotated_degrees(
            -180
        ).normalized()  # clockwise is positive
        left_tail_wing_drag_force = (
            left_tail_wing_drag_direction * left_tail_wing_drag_force_magnitude
        )

        self.rigidbody.apply_force_at_local_point(
            2 * left_tail_wing_drag_force, left_tail_wing_local_coordinate
        )

    def ApplyThrustForce(self):
        # Target propeller rpm calculation

        # Propeller force calculation
        # float rho, tp, pr;
        # for h < 11000, Tp = 15.04 - 0.00649 * h, p = 101.29 * ((Tp + 273.1) / 288.08)^5.256
        # for h < 25000, Tp = -56.46, p = 22.65 * exp(1.73 - 0.000157h)
        # for h > 25000, Tp = -131.21 + 0.00299 * h, p = 2.488 * ((Tp + 273.1) / 216.6))^-11.388
        position = self.rigidbody.position
        x = position.x
        y = self.y_offset - position.y
        if y <= 11000.0:
            tp = 15.04 - 0.00649 * y
            pr = 101.29 * (((tp + 273.1) / 288.08) ** 5.256)
        elif y <= 25000.0:
            tp = -56.46
            pr = 22.65 * (math.e ** (1.73 - 0.000157 * y))
        else:
            tp = -131.21 + 0.00299 * y
            pr = 2.488 * (((tp + 273.1) / 216.6) ** -11.388)
        rho = pr / (0.2869 * (tp + 273.1))

        # compute velocity along body x axis
        velocity = self.rigidbody.velocity
        x_velocity = velocity.projection(self.rigidbody.rotation_vector)

        thrust_force = (
            (-0.125 * x_velocity.length / (self.propeller_rpm / 60.0 * 1.9304) + 0.1)
            * rho
            * ((self.propeller_rpm / 60.0) ** 2.0)
            * (1.9304**4.0)
        )

        self.rigidbody.apply_force_at_local_point(
            thrust_force * self.rigidbody.rotation_vector.normalized()
        )

        # print(f"thrust force: {thrust_force}")

    def Draw(self):
        self.window.fill("black")
        self.space.debug_draw(self.draw_options)
        pygame.display.update()
