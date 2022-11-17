from types import SimpleNamespace

import numpy as np

from utils import set_up_visdom

SAMPLE_RATE = 44100
DURATION = 30
F = 1000


if __name__ == '__main__':
    H = SimpleNamespace()
    H.visdom_port = 8097
    H.visdom_server = 'localhost'
    vis = set_up_visdom(H)

    for step in range(200):

        [[c_p, c_d], [v_p, v_d]] = [[1, 1], [0.5, 0.5]] #evaluate_unconditional(train_loader, ema_sampler if H.ema else sampler, H)
        print(c_p, c_d, v_p, v_d)
        vis.line(
            np.array([c_p]),
            np.array([step]),
            win='Pitch',
            update='append',
            name='consistency',
            opts=dict(title='Con, Var Pitch')
        )
        vis.line(
            np.array([v_p]),
            np.array([step]),
            win='Pitch',
            update='append',
            name='variance',
            opts=dict(title='Con, Var Pitch')
        )
        vis.line(
            np.array([c_d]),
            np.array([step]),
            win='Duration',
            update='append',
            name='consistency',
            opts=dict(title='Con, Var Duration')
        )
        vis.line(
            np.array([v_d]),
            np.array([step]),
            win='Duration',
            update='append',
            name='variance',
            opts=dict(title='Con, Var Duration')
        )