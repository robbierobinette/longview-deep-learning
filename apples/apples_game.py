from ship import Ship
from apple import Apple
from xypoint import XYPoint
import random
from random import randrange
from sensor_map import SensorMap
from policy_gradient import PolicyGradientOptimizer, QModel
import collections
import sys
from timings import Timings
import time
import tensorflow as tf
from math import pi


# this is the bulk of the game, Note that it does not do any drawing.  All drawing is managed
# by interactive_apples.py so that the rest of the game can be run in the cloud on hardware that
# does not have a physical display



class ApplesGame():
    def __init__(self, parameters: {}, session: tf.Session):
        width = parameters.get('width', 1000)
        height = parameters.get('height', 1000)

        self.session = session
        self.model_type = parameters.get('model_type', 3)
        self.width = parameters.get('width', 1000)
        self.height = parameters.get('height', 1000)
        self.n_apples = parameters.get('n_apples', 100)
        self.n_eyes = parameters.get('n_eyes', 9)
        self.stop = parameters.get('stop', 10000)
        self.speed = parameters.get('speed', 3)
        self.gamma = parameters.get('gamma', .9)
        self.action_buffer = parameters.get('action_buffer', 128)
        self.memory_size = parameters.get('memory_size', 1024 * 10)
        self.eye_length = parameters.get('eye_length', 200)
        self.sensor_levels = parameters.get('sensor_levels', 10)
        self.auto = parameters.get('auto', True)
        self.score_buffer = parameters.get('score_buffer', 10240)
        self.exploration = parameters.get('exploration', .10)
        self.label = parameters.get('label', 'no_label')

        self.repeat = False

        self.layer_size = parameters.get('layer_size')
        self.n_layers = parameters.get('n_layers')
        self.block_size = parameters.get('block_size', 128)
        self.learn_rate = parameters.get('learn_rate', 0.00001)
        self.finished = False
        self.res_blocks = parameters.get('res_blocks', 4)
        self.res_layers = parameters.get('res_layers', 3)
        self.dropout = parameters.get('dropout', False)
        self.layer_norm = parameters.get('layer_norm', False)
        self.tensorboard_dir = parameters.get('tensorboard_dir', "tf-logs")
        self.sensor_width = parameters.get('sensor_width', pi / 2)

        self.width = self.width
        self.height = self.height
        self.board_width = self.width
        self.board_height = self.height / 2
        self.timings = Timings()
        self.status = 'starting'
        self.snap()

        self.ship = Ship(XYPoint(self.board_width / 2, self.board_height / 2, ), 0, self.board_width, self.board_height,
                         self.n_eyes, self.sensor_levels, self.eye_length, self.sensor_width)
        self.red = 'red'
        self.green = 'green'
        self.up = False
        self.right = False
        self.left = False
        self.score = 0
        self.peak_score = 0
        self.sensor = SensorMap(self.ship)
        self.step = 0
        self.n_actions = 3

        self.model = self.build_model()
        self.policy = PolicyGradientOptimizer(self.model, self.sensor.size, 3, self.memory_size, self.action_buffer,
                                              self.gamma, self.exploration, self.timings)
        self.scores = collections.deque()
        self.reset_apples()

        self.adjust_score(0)
        self.sensor.update(self.apples.values())
        self.next_state = []

    def snap(self):
        import os
        snapdir = "snapshots/" + self.label
        print("snapdir is %s" % snapdir)
        os.system("mkdir -p " + snapdir )
        os.system("cp *.py %s" % snapdir)

    def reset_apples(self):
        self.apples = {}
        for i in range(self.n_apples):
            self.add_apple(True)
            self.add_apple(False)

    def build_model(self) -> QModel:
            return QModel(self.session,
                           self.label,
                           self.n_actions,
                           self.sensor.size,
                           self.layer_size,
                           self.dropout,
                           self.res_blocks,
                           self.res_layers,
                           self.n_layers,
                           self.layer_norm,
                           self.learn_rate,
                           self.tensorboard_dir)


    def add_apple(self, is_red: bool):
        xy = XYPoint(randrange(self.board_width), randrange(self.board_height))
        while (xy.x, xy.y) in self.apples:
            xy = XYPoint(randrange(self.board_width), randrange(self.board_height))

        apple = Apple(xy, 10, is_red)
        self.apples[(apple.xy.x, apple.xy.y)] = apple

    def update(self, dt):
        pass


    def manual_step(self):
        reward = 0
        action = -1
        if self.up:
            action = 0
        if self.right:
            action = 1
        if self.left:
            action = 2

        if not self.repeat:
            self.up = self.right = self.left = False

        if action != -1:
            self.take_action(action)
            state = self.sensor.as_input()
            action, next_state = self.policy.get_action(state)
            self.next_state = next_state

    def take_action(self, action: int) -> float:
        reward = 0
        if action == 0:
            reward += self.ship.move_forward()
        elif action == 1:
            reward += self.ship.move_right()
        elif action == 2:
            reward += self.ship.move_left()

        t = time.time()
        reward += self.check_for_collision()
        self.timings.add("check_for_collision", time.time() - t)

        t = time.time()
        self.sensor.update(self.apples.values())
        self.timings.add("ship.update_screen", time.time() - t)

        self.step += 1
        self.adjust_score(reward)
        self.model.add_score(self.score_delta(), self.step)
        self.move_random_apple()

        sd = self.score_delta()
        self.status = "score_delta %5.0f peak_score %4.0f memory_size %6d step %6d speed %3d" % (
            sd, self.peak_score, len(self.policy.memory), self.step, self.speed)
        return reward

    def move_random_apple(self):
        for apple in self.apples.values():
            if (random.randrange(0, 10000) == 4):
                del self.apples[(apple.xy.x, apple.xy.y)]
                self.add_apple(apple.ripe())

    def auto_step(self):
        state = self.sensor.as_input()

        # the policy used to return a random action if it was not ready but that
        # does not work well when recovering from a saved game.
        if self.policy.ready():
            action, next_state = self.policy.get_action(state)
        else:
            # randint() is *totally* broken.  things like this should return an element
            # where min <= x < max, i.e. in the range [min, max) not [min, max].  blech.
            action, next_state = random.randint(0, self.n_actions - 1), []

        self.next_state = next_state

        reward = self.take_action(action)

        self.policy.add_action(state, action, reward)
        self.policy.train(1, self.block_size)


    def adjust_score(self, amount: float):
        self.score += amount
        self.scores.appendleft(self.score)
        while len(self.scores) > self.score_buffer:
            self.scores.pop()
        sd = self.score_delta()

        if (sd > self.peak_score):
            self.peak_score = sd

        self.status = "score_delta %5.0f peak_score %4.0f score %6.0f, memory_size %6d step %6d speed %3d" % (
            sd, self.peak_score, self.score, len(self.policy.memory), self.step, self.speed)



    def check_for_collision(self) -> int:
        reward = 0
        for apple in self.apples.values():
            if self.ship.can_eat(apple):
                del self.apples[(apple.xy.x, apple.xy.y)]
                if apple.red:
                    reward += 1
                else:
                    reward -= 1
                self.add_apple(apple.ripe())
        return reward


    def score_delta(self) -> int:
        if len(self.scores) == 0:
            return self.score
        else:
            return self.score - self.scores[-1]

    def finish(self):
        self.model.close_session()
        score_delta = self.score_delta()
        self.status = "%s: final_score %6.0f score_delta %6.0f peak %6.0f step %6.0f" % (self.label, self.score, score_delta, self.peak_score, self.step)
        print(self.status)

        self.finished = True
        self.policy.timings.print()
        sys.stdout.flush()


    def run_no_window(self):
        i = 0
        start = time.time()
        for i in range(self.stop):
            self.auto_step()
            if (i > 0 and i % 1000 == 0):
                now = time.time()
                print("%6.2f %s" % (now - start, self.status))
                start = now


        self.finish()

