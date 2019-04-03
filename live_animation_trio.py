#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use Matplotlib to generate live animation of cat dynamics.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as img
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox, AnchoredOffsetbox)
from matplotlib import animation
import cat_dynamics
import time
import sys
reload(sys)
sys.setdefaultencoding('utf8')
# np.random.seed(0)
plt.rcParams['toolbar'] = 'None'


def update(data):
    pos, vel = data

    # Update cats:
    for i in range(len(abbox)):
        scaledpos = scalefactor * (.9 * pos[i])
        abbox[i].xybox = scaledpos
        if vel[i][1] > 0:
            abbox[i].offsetbox = ibf[i]  # Face forwards
        else:
            abbox[i].offsetbox = ibb[i]  # Face backwards

    # Update labels:
    for i in range(ncat):
        shift = np.array([0, .1])
        scaledpos = scalefactor * (.9 * (pos[i] + shift))
        labels[i].set_position(scaledpos)

    # Update background
    if time.time() - starttime > t1:
        im.set_data(bg2)

    if time.time() - starttime > t2:
        im.set_data(bg3)

    if time.time() - starttime > t3:
        im.set_data(bg4)
        im.set_zorder(101)

    if time.time() - starttime > t4:
        exit()
    return abbox,


def data_gen():
    while True:
        for _ in range(nlog):
            pos[:], vel[:] = cat_dynamics.timestep_underdamped(
                pos, vel, gamma, T, rcut, L, dt, sigma, epsilon)
        yield pos, vel


if __name__ == '__main__':
    # Define simulation parmeters and initialize dynamics:
    names = """alois cory na-young""".split()
    ncat = len(names)       # Number of cats
    L = 2.5                 # Box length
    gamma = .05              # Friction coefficient
    T = 100000.
    sigma = .3
    epsilon = .5
    dt = .0001
    nlog = 100
    rcscale = .05           # Scale factor determining rcut
    rcut = rcscale * L      # Cutoff radius
    t0 = 1                  # Initial frozen frame to get oriented
    t1 = 60                 # Ocean
    t2 = 120                # Field and mac n cheese
    t3 = 180                # Outer space
    t4 = 190                # End

    # Initialize:
    pos = np.array([[0, 1], [.5, 0], [1, 1]])
    pos += L / 2.
    vel = np.zeros(shape=(ncat, 2))
    starttime = time.time()

    # Initialize matplotlib figure
    fig, ax = plt.subplots(facecolor='black', figsize=(8, 8))
    bg1 = img.imread('f/bgocean.jpg')
    bg2 = img.imread('f/bg2.png')
    bg3 = img.imread('f/bgspace.jpg')
    bg4 = img.imread('f/endpage.png')
    im = ax.imshow(bg1, alpha=1.0, zorder=0)

    # Add cats:
    ibb = [OffsetImage(img.imread('icons/{}b.png'.format(i)), zoom=1.3) for
           i in range(ncat)]
    ibf = [OffsetImage(img.imread('icons/{}f.png'.format(i)), zoom=1.3) for
           i in range(ncat)]
    abbox = [AnnotationBbox(ib, [0, 0], xycoords='data', frameon=False) for
             ib in ibb]
    for ab in abbox:
        ax.add_artist(ab)

    # Add labels:
    labels = []
    for i in range(ncat):
        bbox = {'fc': '0.8', 'pad': 0, 'facecolor': 'white',
                'alpha': 0.8, 'pad': 0}
        props = {'ha': 'center', 'va': 'bottom', 'bbox': bbox, 'color': 'k',
                 'fontsize': 24, 'family': 'monospace', 'weight': 'bold',
                 'zorder': 100}
        labels.append(ax.text(0, 0, names[i], props))

    scalefactor = ax.get_xlim()[1] / L

    # Specify animation settings
    ani = animation.FuncAnimation(
        fig, update, data_gen, blit=False, interval=5)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()
    ax.axis('off')  # Turn off ugly border
    plt.show()
