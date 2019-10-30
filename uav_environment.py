import numpy as np


class Environment:
    def __init__(self, vel, steps, turn, cap_dist, cap_angle, start_dev):
        self.GREEDY_VEL = vel           # greedy shooter velocity
        self.MAX_STEPS = steps          # maximum number of time steps per game
        self.MAX_TURN = turn            # maximum instantaneous turn angle in radians
        self.CAPTURE_DIST = cap_dist    # max distance between agent and opponent for successful capture
        self.CAPTURE_ANGLE = cap_angle  # max angle from agent heading to opponent for successful capture
        self.START_DEV = start_dev      # one standard dev of distance in x and y directions between agents at start
        self.uav_state = None           # numpy array of x-position, y-position, heading angle in [0, 2*pi]
        self.greedy_state = None        # numpy array of x-position, y-position, heading angle in [0, 2*pi]
        self.time = None                # current time step starting at 0 during initialization
        self.history = {}               # information about the game indexed by time step

    def reset(self, uav_state=None, greedy_state=None):
        self.reset_uav(uav_state)
        self.reset_greedy(greedy_state)
        if not self.is_good_start():
            return self.reset()
        else:
            self.time = 0
            self.update_history(0, False, None)
            return self.construct_obs()

    def reset_uav(self, uav_state):
        if uav_state is not None:
            self.uav_state = uav_state
        else:
            uav_x = np.random.normal(0, self.START_DEV)
            uav_y = np.random.normal(0, self.START_DEV)
            uav_angle = np.random.uniform(0, 2 * np.pi)
            self.uav_state = np.array([uav_x, uav_y, uav_angle])

    def reset_greedy(self, greedy_state=None):
        if greedy_state is not None:
            self.greedy_state = greedy_state
        else:
            greedy_angle = np.random.uniform(0, 2 * np.pi)
            self.greedy_state = np.array([0, 0, greedy_angle])

    def construct_obs(self):
        """Computes position and heading angle of uav from perspective of greedy shooter to exploit translational
           and rotational symmetry. Greedy shooter is assumed to be located at the origin with heading at 0 radians."""
        x = self.uav_state[0] - self.greedy_state[0]
        y = self.uav_state[1] - self.greedy_state[1]
        angle = -1 * self.greedy_state[2]
        rot_x, rot_y = self.rotate_about_origin(x, y, angle)
        rot_angle = (self.uav_state[2] + angle) % (2 * np.pi)
        return np.array([rot_x, rot_y, rot_angle])

    def update_history(self, reward, done, end_result):
        self.history[self.time] = (np.concatenate((self.uav_state, self.greedy_state)), reward, done, end_result)

    def step(self, uav_action, uav_vel):
        greedy_action = self.calculate_greedy_angle()
        self.greedy_state = self.update_state(greedy_action, self.greedy_state, self.GREEDY_VEL)
        self.uav_state = self.update_state(uav_action, self.uav_state, uav_vel)

        self.time += 1
        uav_can_fire, greedy_can_fire = self.check_both_can_fire()
        time_expired = self.time >= self.MAX_STEPS
        reward, end_result = self.determine_reward(uav_can_fire, greedy_can_fire, time_expired)
        done = uav_can_fire or greedy_can_fire or time_expired
        self.update_history(reward, done, end_result)
        return self.construct_obs(), reward, done, self.history

    @staticmethod
    def determine_reward(uav_caught, greedy_caught, time_expired):
        if greedy_caught:
            return 0, "GREEDY_WIN"
        else:
            if uav_caught:
                return 1, "UAV_WIN"
            else:
                if time_expired:
                    return 0, "DRAW"
                else:
                    return 0, None

    def calculate_greedy_angle(self):
        target_angle = np.arctan2(self.uav_state[1] - self.greedy_state[1], self.uav_state[0] - self.greedy_state[0])
        greedy_angle = self.greedy_state[2]
        diff = self.angle_diff(target_angle, greedy_angle)
        return np.clip(diff, -1 * self.MAX_TURN, self.MAX_TURN)

    def update_state(self, angle, old_state, velocity):
        new_angle = (old_state[2] + angle) % (2 * np.pi)
        new_x = old_state[0] + velocity * np.cos(new_angle)
        new_y = old_state[1] + velocity * np.sin(new_angle)
        return np.array([new_x, new_y, new_angle])

    def is_good_start(self):
        uav_can_fire = self.is_able_to_fire(self.uav_state, self.greedy_state)
        greedy_can_fire = self.is_able_to_fire(self.greedy_state, self.uav_state)
        return not (uav_can_fire or greedy_can_fire)

    def check_both_can_fire(self):
        uav_can_fire = self.is_able_to_fire(self.uav_state, self.greedy_state)
        greedy_can_fire = self.is_able_to_fire(self.greedy_state, self.uav_state)
        return uav_can_fire, greedy_can_fire

    def is_able_to_fire(self, agent, opponent):
        return self.is_within_capture_dist(agent, opponent) and self.is_within_capture_angle(agent, opponent)

    def is_within_capture_dist(self, agent, opponent):
        return self.position_distance(agent[0:2], opponent[0:2]) < self.CAPTURE_DIST

    def is_within_capture_angle(self, agent, opponent):
        target_angle = np.arctan2(opponent[1] - agent[1], opponent[0] - agent[0])
        return np.abs(self.angle_diff(agent[2], target_angle)) < self.CAPTURE_ANGLE

    @staticmethod
    def rotate_about_origin(x, y, angle):
        new_x = np.cos(angle) * x - np.sin(angle) * y
        new_y = np.sin(angle) * x + np.cos(angle) * y
        return np.array([new_x, new_y])

    @staticmethod
    def angle_diff(x, y):
        """Calculates the difference of angle x minus angle y in range [-pi, pi]."""
        return np.arctan2(np.sin(x - y), np.cos(x - y))

    @staticmethod
    def position_distance(pos_x, pos_y):
        return np.sqrt(np.sum(np.square(pos_x - pos_y)))
