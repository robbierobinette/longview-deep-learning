import gym
from policy_gradient import QModel, PolicyGradientOptimizer
import tensorflow as tf
from timings import Timings
import os


def main():
    # this makes a simple atari pong game.  the -ram- means that we are seeing the 128 bytes of RAM in the
    # atari computer rather than the actual pixels.
    p = gym.make('Pong-ram-v0')
    p._max_episode_steps = 1000000

    # set up a few things:  We need to get the initial state so we can figure out it size and make the
    # neural network match it properly.
    initial_state = p.reset()
    state_size: int = initial_state.size
    n_actions: int = p.action_space.n

    # set up a few initial things about tensorflow.
    session = tf.Session()
    global_step_tensor = tf.Variable(0, trainable= False, name = 'global_step')
    increment_global_step = global_step_tensor.assign_add(1)

    # find an empty place to store output for tensorboard (a web app to look at results.)
    run_counter = 0
    tensorboard_dir = "../tensorboard/pong-simple-%03d" % run_counter
    while os.path.exists(tensorboard_dir):
        run_counter += 1
        tensorboard_dir = "../tensorboard/pong-simple-%03d" % run_counter

    print("tensorboard_dir: %s" % tensorboard_dir)

    # build a simple model.
    model = QModel(session, "pong-model",
                   n_actions = n_actions,
                   state_size = state_size,
                   layer_size = 512,
                   dropout = True,
                   res_blocks = 0,
                   res_layers = 0,
                   layers = 3,
                   layer_norm = True,
                   learn_rate = .00001,
                   tensorboard_dir = tensorboard_dir)

    # this is a little widget to keep track of how much time we spend doing things
    timings = Timings()
    # this is the thing that tracks the State-Action-Reward tuples and trains the network.
    policy_optimizer = PolicyGradientOptimizer(
            model = model,
            input_size = state_size,
            n_actions = n_actions,
            max_memory_size = 1024 * 1024,
            action_buffer_size = 102400,
            gamma = .999,
            exploration = .05,
            timings = timings)

    # this allows us to restart from wherever we left off.  These checkpoints can also be used by
    # "saved_pong.py" to watch how the game is currently playing.
    saver = tf.train.Saver()
    checkpoint_file = "../checkpoints/pong-simple/ckpt"
    saved_checkpoint_path = tf.train.latest_checkpoint("../checkpoints/pong-simple/")
    from_saved_model = False
    if saved_checkpoint_path is not None:
        print("restoring from %s" % saved_checkpoint_path)
        saver.restore(session, saved_checkpoint_path)
        from_saved_model = True

    # this global_step tensor is really just a counter for the current step but I want to be able to
    # recover it after a restart so I store it and increment it within tensorflow.
    global_step = tf.train.global_step(session, global_step_tensor)
    training_started = True

    # this is the outermost loop.  it will never exit.  each new game will start at the top of this loop.
    while True:
        initial_state = p.reset()
        done = False
        summed_rewards = 0
        raw_score = 0
        steps = 0.0
        state = initial_state.reshape([1,128]) # this starts out as a vector and needs to be a [1x128] matrix.
        # this loop runs one game.  At the end of the game the step function returns done=True
        while not done:
            steps += 1.0
            # if we have loaded a saved model or if we have begun training then we can use the model to choose
            # an action.  Otherwise, pick one at random.
            if from_saved_model or policy_optimizer.ready():
                action, _ = policy_optimizer.get_action(state)
            else:
                action = p.action_space.sample()

            new_state, reward, done, info = p.step(action)
            raw_score += reward
            # Using just the reward from the game wasn't very effective.  It did not learn to play well.
            # after thinking about it I considered a situation where the game plays a long point and loses at the end.
            # It may have properly returned the ball 10 or 20 times but those *all* get told that they messed up.
            # I decided that only the actions before missing the ball should get a negative reward and everything else
            # should be positive.  This worked great.  Since it takes about 50 frames (steps) for the ball to go from
            # the left edge to the right edge I put in a rule that if it loses the points the last 50 frames get a
            # negative reward and everything else gets +.5.
            #
            # Winning points get +1 for all frames.
            if reward < 0:
                count = 0
                for sar in policy_optimizer.action_buffer:
                    if (count < 50):
                        sar.add_reward(-1)
                    else:
                        sar.add_reward(.5)
                    policy_optimizer.add_to_memory(sar)
                    count += 1
                policy_optimizer.action_buffer.clear()
            elif reward > 0:
                policy_optimizer.add_action(state, action, reward + steps / 400)
                policy_optimizer.flush_action_buffer()
            elif reward == 0:
                policy_optimizer.add_action(state, action, reward)

            # This is where the magic happens.  randomly select 32 of the actions we've taken and
            # see what we can learn about them.
            policy_optimizer.train(1, 32)

            state = new_state.reshape([1,128])
            summed_rewards += reward

        # increment the global step and save our progress after each complete game (not point).
        session.run(increment_global_step)
        global_step = tf.train.global_step(session, global_step_tensor)
        print("iteration %6d steps %6d, training_score: % 6.2f raw_score % 6.2f" %
              (global_step, steps, summed_rewards, raw_score))
        saver.save(session, checkpoint_file, global_step = global_step)

        # this logs things to tensorboard
        model.add_score(steps, global_step)
        # if we are a multiple of 10 steps print out the timings.  I know what these are by now
        # but it's interesting to see.
        if ((global_step + 1) % 10 == 0):
            timings.print()
            timings.reset()

if __name__ == "__main__":
    main()
