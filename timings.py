import time
class Timings(object):
    def __init__(self):
        self.timings = {}

    def add(self, label: str, delta: float):
        self.timings[label] = self.timings.get(label, 0.0) + delta

    def print(self):
        for k in self.timings.keys():
            print("%20s %.3f" % (k, self.timings[k]))

    def reset(self):
        self.timings = {}


