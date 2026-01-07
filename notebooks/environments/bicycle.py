# define bicycle environment here
import math

import numpy as np


class Bicycle:
    def __init__(self):
        self.action_set = [
            np.array(pair)
            for pair in [[0.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [0.0, -0.02], [0.0, 0.02]]
        ]

        # observable state variables
        self.theta = 0.0  # angle of the handlebar
        self.theta1 = 0.0  # first derivative of theta
        self.omega = 0.0  # vertical angle of the bicycle
        self.omega1 = 0.0  # first derivative of omega
        self.xb = 0.0  # back wheel position on x
        self.yb = 0.0  # back wheel position on y
        self.xf = 0.0  # front wheel position on x
        self.yf = 0.0  # front wheel position on y

        # feature variables
        self.psi_goal = 0.0  # not really an angle, is roughly proportional to the angle between bike and goal
        self.dist_to_goal = 0.0  # distance between front wheel and goal's boundary
        self.heading = 0.0  # angle between the bike and the y-axis
        self.angle_to_goal = 0.0  # angle between bike and goal

        # bike parameters
        self.dt  = 0.01    # control period
        self.v   = 10./3.6 # bike velocity
        self.g   = 9.81    # gravity
        self.dCM = 0.3     # distance between bicycle and cyclist centers of mass
        self.c   = 0.66    # horizontal distance between front wheel ground contact and center of mass
        self.h   = 0.94    # height of bicycle+cyclist center of mass
        self.Mc  = 15.0    # bicycle mass
        self.Md  = 1.7     # tyre mass
        self.Mp  = 60.0    # cyclist mass
        self.M   = self.Mc + self.Mp # bicycle+cyclist mass
        self.r   = 0.34    # tyre radius
        self.sigma_dot = self.v / self.r # tyre angular velocity
        self.Ibike = 13.0 * self.Mc * self.h**2 / 3.0 + self.Mp * (self.h + self.dCM)**2 # Moment of inertia for bike+cyclist
        self.Idc   = self.Md * self.r**2 # Moment of inertia for the type along the rotation axis
        self.Idv   = 3.*self.Md * self.r**2 / 2. # Moment of inertia for the tyre along the velocity axis
        self.Idl   = 1.*self.Md * self.r**2 / 2. # Moment of inertia for the type along the handlebar axis
        self.l   = 1.11    # distance between front wheel and back wheel ground contact points

        # goal
        self.x_goal = 1000.0
        self.y_goal = 0.0
        self.radius_goal = 10.0

        # action noise
        self.max_displacement_noise = 0.02

        self.reset()
        return

    def reset(self):
        self.heading = np.random.uniform(-np.pi, np.pi)

        self.theta = np.random.uniform(-np.pi / 100.0, np.pi / 100.0)
        self.theta1 = np.random.uniform(-0.1, 0.1)
        self.omega = np.random.uniform(-np.pi / 1200.0, np.pi / 1200.0)
        self.omega1 = np.random.uniform(-0.05, 0.05)
        self.xb = 0.0
        self.yb = 0.0
        self.xf = self.xb - self.l * math.sin(self.heading)
        self.yf = self.yb + self.l * math.cos(self.heading)

        self.psi_goal = self.calc_angle_to_goal_legacy()
        self.angle_to_goal = self.calc_angle_to_goal()
        self.dist_to_goal = self.calc_distance_to_goal()
        self.heading = self.calc_heading()
        return np.array(
            [
                self.theta,
                self.theta1,
                self.omega,
                self.omega1,
                self.xb,
                self.yb,
                self.xf,
                self.yf,
                self.psi_goal,
                self.dist_to_goal,
                self.heading,
                self.angle_to_goal,
            ]
        ), None

    def calc_distance_to_goal(self):
        """distance between front wheel and goal circle boundary"""
        dist_to_center = (self.xf - self.x_goal) ** 2 + (self.yf - self.y_goal) ** 2
        dist_to_center = math.sqrt(dist_to_center)
        return np.max([0.0, dist_to_center - self.radius_goal])

    def calc_angle_to_goal_legacy(self):
        """from Lagoudakis's code

        Original comment:
        These angles are neither in degrees nor radians, but something
        strange invented in order to save CPU-time. The measure is arranged the
        same way as radians, but with a slightly different negative factor.

        Say, the goal is to the east.
        If the agent rides to the east then  temp = 0
        - " -          - " -   north              = -1
        - " -                  west               = -2 or 2
        - " -                  south              =  1
        """
        temp = (self.xf - self.xb) * (self.x_goal - self.xf) + (self.yf - self.yb) * (
            self.y_goal - self.yf
        )
        scalar = temp / (
            self.l
            * math.sqrt((self.x_goal - self.xf) ** 2 + (self.y_goal - self.yf) ** 2)
        )
        tvaer = (-self.yf + self.yb) * (self.x_goal - self.xf) + (self.xf - self.xb) * (
            self.y_goal - self.yf
        )
        return scalar - 1 if tvaer <= 0 else abs(scalar - 1)

    def calc_angle_to_goal(self):
        """angle between bicycle axis and back wheel-goal axis"""
        BF = np.array([self.xf - self.xb, self.yf - self.yb])
        BFunit = BF / np.linalg.norm(BF)
        BG = np.array([self.x_goal - self.xb, self.y_goal - self.yb])
        BGunit = BG / np.linalg.norm(BG)
        cosinus = np.dot(BFunit, BGunit)
        sign_sinus = np.sign(
            (self.xf - self.xb) * (self.yb - self.y_goal)
            + (self.yf - self.yb) * (self.x_goal - self.xb)
        )
        if cosinus > 1.0:
            cosinus = 1.0
        elif cosinus < -1.0:
            cosinus = -1.0
        return sign_sinus * math.acos(cosinus)

    def calc_heading(self):
        """angle between bicycle and y-axis"""
        diff_y = self.yf - self.yb
        diff_x = self.xb - self.xf
        if diff_x == 0.0 and diff_y < 0:
            angle = np.pi
        else:
            if diff_y > 0:
                angle = math.atan(diff_x / diff_y)
            else:
                angle = np.sign(diff_x) * np.pi / 2.0 - math.atan(diff_y / diff_x)
        return angle

    def step(self, action_index):
        action = self.action_set[action_index]
        torque = action[0]  # torque applied to handlebar
        displacement = action[1]  # lateral displacement of cyclist
        displacement += self.max_displacement_noise * np.random.uniform(-1.0, 1.0)

        # turn radii of front tyre, back tyre and center of mass
        if self.theta == 0.0:
            rCM = rf = rb = (
                1e9  # just a large value to represent in immense turn radius if the bike is going straight
            )
        else:
            tantheta = math.tan(self.theta)
            sintheta = math.sin(self.theta)
            rCM = math.sqrt((self.l - self.c) ** 2 + self.l**2 / (tantheta**2))
            rf = self.l / abs(sintheta)
            rb = self.l / abs(tantheta)

        # dynamics equations
        # tilt angle between vertical and bicycle+cyclist center of mass
        phi = self.omega + math.atan(displacement / self.h)
        # second order derivative of omega
        omega2 = (
            self.h * self.M * self.g * math.sin(phi)
            - math.cos(phi)
            * (
                self.Idc * self.sigma_dot * self.theta1
                + np.sign(self.theta)
                * self.v**2
                * (self.Md * self.r * (1.0 / rf + 1.0 / rb) + self.M * self.h / rCM)
            )
        ) / self.Ibike
        # second order derivative of theta
        theta2 = (torque - self.Idv * self.omega1 * self.sigma_dot) / self.Idl

        # integration
        new_omega = self.omega + self.omega1 * self.dt
        new_omega1 = self.omega1 + omega2 * self.dt
        new_theta = self.theta + self.theta1 * self.dt
        new_theta1 = self.theta1 + theta2 * self.dt

        # handlebar cannot turn more than 80 degrees
        if abs(new_theta) > 1.3963:
            new_theta = np.sign(new_theta) * 1.3963

        # update position of tyres
        # front tyre
        temp = self.v * self.dt / (2.0 * rf)
        if temp > 1:
            temp = np.sign(self.heading + self.theta) * np.pi / 2.0
        else:
            temp = np.sign(self.heading + self.theta) * math.asin(temp)
        self.xf += self.v * self.dt * (-math.sin(self.heading + self.theta + temp))
        self.yf += self.v * self.dt * math.cos(self.heading + self.theta + temp)
        # back tyre
        temp = self.v * self.dt / (2.0 * rb)
        if temp > 1:
            temp = np.sign(self.heading) * np.pi / 2.0
        else:
            temp = np.sign(self.heading) * math.asin(temp)
        self.xb += self.v * self.dt * (-math.sin(self.heading + temp))
        self.yb += self.v * self.dt * math.cos(self.heading + temp)

        # after a while the bike deforms due to rounding errors
        # if that happens, just drag the back wheel so that the bike does not tear down
        bike_length = math.sqrt((self.xf - self.xb) ** 2 + (self.yf - self.yb) ** 2)
        length_error = self.l - bike_length
        if abs(length_error) > 1e-3:
            self.xb += (self.xb - self.xf) * (length_error) / bike_length
            self.yb += (self.yb - self.yf) * (length_error) / bike_length

        # update heading, angle to goal and distance to goal
        new_psi_goal = self.calc_angle_to_goal_legacy()
        new_heading = self.calc_heading()
        new_dist_to_goal = self.calc_distance_to_goal()
        new_angle_to_goal = self.calc_angle_to_goal()

        # reward model
        done = False
        reward = (self.omega * 15 / np.pi) ** 2 - (new_omega * 15 / np.pi) ** 2
        if abs(new_omega) > np.pi / 15:
            done = True
        else:
            reward += 0.01 * (self.dist_to_goal - new_dist_to_goal)

        # update state
        self.theta = new_theta
        self.theta1 = new_theta1
        self.omega = new_omega
        self.omega1 = new_omega1
        self.psi_goal = new_psi_goal
        self.heading = new_heading
        self.dist_to_goal = new_dist_to_goal
        self.angle_to_goal = new_angle_to_goal
        state = np.array(
            [
                self.theta,
                self.theta1,
                self.omega,
                self.omega1,
                self.xb,
                self.yb,
                self.xf,
                self.yf,
                self.psi_goal,
                self.dist_to_goal,
                self.heading,
                self.angle_to_goal,
            ]
        )

        return state, reward, done, None, None
