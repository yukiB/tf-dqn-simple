import os
import numpy as np


class Ball:
    def __init__(self, col):
        self.col = col
        self.row = 0

    def update(self):
        self.row += 1

    def isDroped(self, n_rows):
        return True if self.row == n_rows else False


class AvoidBall:
    def __init__(self):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.screen_n_rows = 8
        self.screen_n_cols = 8
        self.player_length = 3
        self.enable_actions = (0, 1, 2)
        self.frame_rate = 5
        self.ball_post_interval = 4
        self.ball_past_time = 0
        self.past_time = 0
        self.balls = []

        # variables
        self.reset()

    def update(self, action):
        """
        action:
            0: do nothing
            1: move left
            2: move right
        """
        if self.terminal:
            self.reset()
            
        # update player position
        if action == self.enable_actions[1]:
            # move left
            self.player_col = max(0, self.player_col - 1)
        elif action == self.enable_actions[2]:
            # move right
            self.player_col = min(self.player_col + 1, self.screen_n_cols - self.player_length)
        else:
            # do nothing
            pass

        # update ball position
        for b in self.balls:
            b.row += 1

        if self.ball_past_time == self.ball_post_interval:
            self.ball_past_time = 0
            self.balls.append(Ball(np.random.randint(self.screen_n_cols)))
        else:
            self.ball_past_time += 1

        self.past_time += 1
        if self.past_time > 100:
            self.terminal = True

        # collision detection
        self.reward = 0
        self.terminal = False
        if self.balls[0].row == self.screen_n_rows - 1:
            if self.player_col <= self.balls[0].col < self.player_col + self.player_length:
                # catch
                self.reward = -1
                self.terminal = True
            else:
                # drop
                self.reward = +1


        for b in self.balls:
            if b.isDroped(self.screen_n_rows):
                self.balls.pop(0)

    def draw(self):
        # reset screen
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols))

        # draw player
        self.screen[self.player_row, self.player_col:self.player_col + self.player_length] = 1

        # draw ball
        for b in self.balls:
            self.screen[b.row, b.col] = 1

    def observe(self):
        self.draw()
        return self.screen, self.reward, self.terminal

    def execute_action(self, action):
        self.update(action)

    def reset(self):
        # reset player position
        self.player_row = self.screen_n_rows - 1
        self.player_col = np.random.randint(self.screen_n_cols - self.player_length)

        # reset ball position
        self.balls = []
        self.balls.append(Ball(np.random.randint(self.screen_n_cols)))

        # reset other variables
        self.reward = 0
        self.terminal = False