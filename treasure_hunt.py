#!/usr/bin/env python
"""
Use Matplotlib to generate live animation of cat dynamics.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as img
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import animation
import cat_dynamics
import time
plt.rcParams['toolbar'] = 'None'


def update(data):
    pos, vel, boat_pos, boat_vel = data
    currtime = time.time() - starttime

    # Update cats:
    for i in range(len(abbox)):
        scaledpos = scalefactor * (.9 * pos[i])
        abbox[i].xybox = scaledpos
        dircode = ''
        if vel[i][1] > 0:
            dircode += 'f'  # Face forwards
        else:
            dircode += 'b'  # Face backward
        if vel[i][0] > 0:
            dircode += 'r'  # Face right
        else:
            dircode += 'l'  # Face left
        abbox[i].offsetbox = imcat[dircode][i]

    # Update boat:
    scaledpos = scalefactor * (.9 * boat_pos[0])
    boatbox.xybox = scaledpos
    nboatframes = 100
    # nboatupdate = 10  # Update every .1 seconds
    # boatid = int((currtime % nboatupdate) * (nboatframes / nboatupdate))
    boatid[0] = (boatid[0] + 2) % nboatframes
    if boat_vel[0][0] < 0:
        boatbox.offsetbox = imboatl[boatid[0]]
    else:
        boatbox.offsetbox = imboatr[boatid[0]]

    # Update labels:
    for i in range(ncat):
        shift = np.array([0, .1])
        scaledpos = scalefactor * (.9 * (pos[i] + shift))
        labels[i].set_position(scaledpos)

    # Update background
    t_elapsed = currtime
    if currtime < tfade:
        imfade.set_alpha(1. - (t_elapsed / tfade))
    if currtime > tfadeout:
        alpha = min(1., (currtime - tfadeout) / tfade)
        imfade.set_alpha(alpha)

    return abbox,


def data_gen():
    while True:
        for _ in range(n_update):
            fpos = np.concatenate([pos, boat_pos])
            fvel = np.concatenate([vel, boat_vel])
            fpos[:], fvel[:] = cat_dynamics.timestep_newton(
                fpos, fvel, L, dt, sigma, epsilon, m)
            cat_dynamics.piano_trio_bc(fpos, fvel, L)
            pos[:], boat_pos[:] = np.split(fpos, [ncat])
            vel[:], boat_vel[:] = np.split(fvel, [ncat])
            # np.sum(vel**2, axis=1)
            vel[:] *= vel_fix / np.linalg.norm(vel)  # Renormalize velocities
            boat_vel[0][1] = 0  # Manually cancel boat y-velocity
            # print pos
        yield pos, vel, boat_pos, boat_vel


if __name__ == '__main__':
    # Define simulation parmeters and initialize dynamics:
    names = """na-young alois cory""".split()
    counter = 0
    ncat = len(names)       # Number of cats
    L = 1.                  # Box length
    m = np.array([1., 1., 1., 100.])
    vel_fix = .04
    sigma = .1
    epsilon = .01
    dt = .01
    n_update = 30            # Update animation every n_update timesteps
    rcscale = .2            # Scale factor determining rcut
    mod = False             # Unused mod from Kranthi class
    rcut = rcscale * L      # Cutoff radius
    tfade = 10
    tfadeout = 180

    # Initialize:
    pos = np.array([[.1, .8],
                    [.8, .7],
                    [.8, .9]])
    boat_pos = np.array([[.4, .55]])
    pos += np.random.random(size=pos.shape) * .01  # Add jitter
    vel = np.array([[.05, 0.],
                    [0., 0.],
                    [0., 0.]])
    boat_vel = np.array([[-0.01, 0.]])
    starttime = time.time()
    boatid = [0]

    # Initialize matplotlib figure
    fig, ax = plt.subplots(facecolor='black', figsize=(8, 8))
    bg1 = img.imread('f/bgocean.jpg')
    im = ax.imshow(bg1, alpha=1.0, zorder=0)
    black = np.zeros_like(bg1)
    # black = np.ones((bg1.shape[0] + 100, bg1.shape[1] + 100))
    imfade = ax.imshow(black, alpha=1.0, zorder=102)

    # Add boat:
    imboatr = [OffsetImage(img.imread(
        'icons/boat/right/boat_00{:03d}.png'.format(i)),
        zoom=1.) for i in range(1, 101)]
    imboatl = [OffsetImage(img.imread(
        'icons/boat/left/boat_00{:03d}.png'.format(i)),
        zoom=1.) for i in range(1, 101)]
    boatbox = AnnotationBbox(imboatr[0], [0, 0],
                             xycoords='data', frameon=False)
    ax.add_artist(boatbox)

    # Add cats:
    imcat = {}
    for l in ['fr', 'br', 'fl', 'bl']:
        imcat[l] = [OffsetImage(
            img.imread('icons/cats/{}{}.png'.format(i, l)), zoom=1.)
            for i in range(ncat)]
    abbox = [AnnotationBbox(ib, [0, 0], xycoords='data', frameon=False) for
             ib in imcat['fr']]
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
        fig, update, data_gen, blit=False, interval=20)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # plt.tight_layout()
    ax.axis('off')  # Turn off ugly border
    plt.show()
