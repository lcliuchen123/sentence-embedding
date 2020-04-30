#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

"""
@Time    : 2020/2/22 17:00
@Author  : Chen Liu
@FileName: grid.py
@Software: PyCharm
 
"""


import matplotlib.pyplot as plt


def noam_scheme(dim, step, lr=0.0005, warmup_steps=4000.):
    """
        Noam scheme learning rate decay
        init_lr: initial learning rate. scalar.
        global_step: scalar.
        warmup_steps: scalar. During warmup_steps, learning rate increases
            until it reaches init_lr.
    """
    # return lr * dim ** 0.5 * min(step * warmup_steps ** -1.5, step ** -0.5)
    return dim ** 0.5 * min(step * warmup_steps ** -1.5, step ** -0.5)


def plot_grid(dim=300, global_step=6520):
    x = [i for i in range(1, global_step)]
    y = [noam_scheme(dim, step) for step in x]
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    plot_grid()
