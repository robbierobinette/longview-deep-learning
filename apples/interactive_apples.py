import sys
sys.path.append("..")
import arcade as a
from apples_game import *
import tensorflow as tf
from visual_ndarray import ArrayVis
from eye import Eye
from mathutils import rotate
import numpy as np
import pyglet


class InteractiveApplesGame(a.Window):
    def __init__(self, parameters: {}, session: tf.Session):

        width = parameters.get('width', 1000)
        height = parameters.get('height', 800)
        a.Window.__init__(self, width, height, title="Apples")
        self.apples_game = ApplesGame(parameters, session)

        self.set_update_rate(1 / 60)
        self.start = time.time()

    def update(self, dt):
        pass

    def on_draw(self):
        game = self.apples_game
        if game.auto:
            for i in range(0, game.speed):
                game.auto_step()
        else:
            game.manual_step()

        self.draw_game()

        if (game.step > 0 and game.step % 1000 == 0):
            now = time.time()
            print("%6.2f %s" % (now - self.start, game.status))
            self.start = now
            game.timings.print()
            game.timings.reset()

        if (self.apples_game.step >= self.apples_game.stop):
            game.finish()
            pyglet.app.event_loop.exit()

    def on_key_press(self, key, modifiers):
        game = self.apples_game
        if key == a.key.UP:
            game.up = True
        if key == a.key.RIGHT:
            game.right = True
        if key == a.key.LEFT:
            game.left = True
        if key == a.key.A:
            game.auto = True
        if key == a.key.M:
            game.auto = False
        if key == a.key.K:
            game.stop += 1024
        if key == a.key.F:
            game.speed += 1
        if key == a.key.S:
            game.speed = 1
        if key == a.key.Q:
            game.speed = 16
            game.auto = True
        if key == a.key.P:
            game.sensor.print()
        if key == a.key.R:
            game.repeat =  not game.repeat

    def on_key_release(self, key, modifiers):
        game = self.apples_game
        if key == a.key.UP:
            game.up = False
        if key == a.key.RIGHT:
            game.right = False
        if key == a.key.LEFT:
            game.left = False

    def draw_apple(self, apple: Apple):
        if apple.red:
            color = a.color.RED
        else:
            color = a.color.GREEN
        a.draw_rectangle_filled(apple.xy.x, apple.xy.y, apple.radius * 1.7, apple.radius * 1.7, color)

    def draw_game(self):
        a.start_render()

        game = self.apples_game

        for apple in game.apples.values():
            self.draw_apple(apple)

        state_width = 200
        state_height = 200

        self.draw_ship(game.ship)

        score_text = a.draw_commands.create_text(game.status, a.color.WHITE)
        a.draw_commands.render_text(score_text, 10, game.board_height + 5)

        colors = [a.color.RED, a.color.GREEN, a.color.BLUE]
        av = ArrayVis(state_width, state_height)

        # self.sensor.draw(XYPoint(20, self.board_height + 100), 200, 200)

        av.draw(game.sensor.screen, XYPoint(20, game.board_height + 100), colors)

        border = 50
        lower_left = state_width + border * 2
        for flat_screen in game.next_state:
            ns = np.reshape(flat_screen, game.sensor.screen.shape)
            av.draw(ns, XYPoint(lower_left, game.board_height + 100), colors)
            lower_left += state_width + border



    def draw_ship(self, ship):

        triangle_points = []
        triangle_points.append(XYPoint(-10, 10))
        triangle_points.append(XYPoint(20, 0))
        triangle_points.append(XYPoint(-10, -10))

        shiftedPoints = []
        for p in triangle_points:
            p = ship.xy + rotate(ship.angle, p)
            shiftedPoints.append((p.x, p.y))

        blue = a.color.BLUE
        a.draw_polygon_filled(shiftedPoints, blue)

        for eye in ship.eyes:
            self.draw_eye(eye)

    def draw_eye(self, eye: Eye):
        a.draw_line(eye.p1.x, eye.p1.y, eye.p2.x, eye.p2.y, a.color.CYAN)



def base_parameters() -> {}:
    parameters = {}
    parameters['width'] = 1000
    parameters['height'] = 800
    parameters['n_apples'] = 100
    parameters['n_eyes'] = 21
    parameters['sensor_levels'] = 10
    parameters['stop'] = 1024 * 1024 * 1024
    parameters['speed'] = 8
    parameters['gamma'] = .9
    parameters['action_buffer'] = 512
    parameters['memory_size'] = 1024 * 100
    parameters['score_buffer'] = 10240
    parameters['eye_length'] = 300
    parameters['auto'] = True
    parameters['model_type'] = 1
    parameters['exploration'] = .10
    parameters['block_size'] = 128
    parameters['learn_rate'] = .00001
    parameters['learn_rate'] = .0001
    parameters['n_layers'] = 3
    parameters['layer_size'] = 1024
    parameters['res_blocks'] = 0
    parameters['res_layers'] = 3
    parameters['dropout'] = False
    parameters['layer_norm'] = True
    parameters['sensor_width'] = pi
    return parameters

def main():
    print("main")
    import tensorflow as tf
    with tf.Session() as session:
        parameters = base_parameters()
        window = InteractiveApplesGame(parameters, session)
        a.run()
        print("close")
        window.close()

if __name__ == "__main__":
    main()
