import sys
sys.path.append("..)

from apples_game import *


def base_parameters() -> {}:
    parameters = {}
    parameters['width'] = 1000
    parameters['height'] = 800
    parameters['n_apples'] = 100
    parameters['n_eyes'] = 9
    parameters['eye_length'] = 1024
    parameters['sensor_levels'] = 10
    parameters['stop'] = 200 * 1024
    parameters['gamma'] = .9
    parameters['action_buffer'] = 512
    parameters['memory_size'] = 1024 * 10
    parameters['score_buffer'] = 10240
    parameters['eye_length'] = 150
    parameters['auto'] = True
    parameters['block_size'] = 128
    parameters['learn_rate'] = .00001
    parameters['layer_size'] = 256
    parameters['n_layers'] = 3
    parameters['learn_rate'] = .0001
    parameters['model_type'] = 1
    parameters['exploration'] = .05
    parameters['res_blocks'] = 0
    parameters['res_layers'] = 3
    parameters['dropout'] = False
    parameters['layer_norm'] = False
    parameters['tensorboard_dir']  = "tf-logs"
    parameters['sensor_width'] = pi / 2
    return parameters


def main():
    all_params = []
    dropout = True
    layer_norm = True
    for res_blocks in [2, 4]:
        for layer_size in [1024]:
            for dropout in [True, False]:
                for layer_norm in [True, False]:
                    parameters = base_parameters()
                    parameters['res_blocks'] = res_blocks
                    parameters['dropout'] = dropout
                    parameters['layer_norm'] = layer_norm
                    parameters['layer_size'] = layer_size

                    label = "layer_size-%d_res_blocks-%d_dropout-%s_layer_norm-%s" % \
                            (layer_size, res_blocks, dropout, layer_norm)
                    parameters['label'] = label

                    all_params.append(parameters.copy())

    for p in all_params:
        r = run_one_set(p)

        # import multiprocessing
        # pool = multiprocessing.Pool()
        # results = pool.map(run_one_set, all_params)
        # for r in results:
        #     print(r)


def run_one_set(parameters: {}) -> str:
    tf.reset_default_graph()
    with tf.Session() as session:
        apples_game = ApplesGame(parameters, session)
        apples_game.run_no_window()
        return apples_game.status


if __name__ == "__main__":
    main()
