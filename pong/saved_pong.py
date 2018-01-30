import gym
import sys
sys.path.append("..")
from policy_gradient import QModelFromSavedFile
import tensorflow as tf
import collections
import numpy as np
import argparse



def append_states(states: collections.deque, count: int) -> np.ndarray:
    # get rid of any extra data we don't need anymore
    while len(states) > count:
        states.pop()
    # if the state deque is not full append zeros, not my favorite thing to do.
    ss = []
    for i in range(count):
        if len(states) > i:
            ss.append(states[i])
        else:
            ss.append(np.zeros(states[0].shape))

    return np.hstack(ss)


def main(args):

    p = gym.make('Pong-ram-v0')
    p._max_episode_steps = 1000000
    session = tf.Session()

    saved_checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)
    print("loading QModel from %s" % saved_checkpoint_path)
    model = QModelFromSavedFile(session, saved_checkpoint_path)

    # sometimes we feed the current state and some past history to the model
    # this determines how much history the model is expecting.
    state_frames = int(model.state_size / p.reset().size)

    while True:
        initial_state = p.reset()
        state_list = collections.deque()
        state_list.appendleft(initial_state.reshape([1, 128]))

        done = False
        summed_rewards = 0
        steps = 0.0
        while not done:
            p.render()
            steps += 1.0
            if (len(state_list) >= state_frames):
                # if we have enough state frames in the buffer to invoke the network:
                state_with_lookback = append_states(state_list, state_frames)
                action, _ = model.get_action(state_with_lookback)
            else:
                action = p.action_space.sample()

            new_state, reward, done, info = p.step(action)
            summed_rewards += reward

            state_list.appendleft(new_state.reshape([1, 128]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "saved_pong:  show game play from saved checkpoint")
    parser.add_argument('--checkpoint_dir', type=str, help="directory containing checkpoint")
    args = parser.parse_args()
    main(args)
