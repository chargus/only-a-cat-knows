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
    pos, thetas, cat_pos, cat_thetas = data
    buff = 0.5  # Hysteresis buffer, to keep cat icons from jittering
    # Update cats:
    for i in range(len(abbox)):
        scaledpos = scalefactor * (1.4 * cat_pos[i] - 1.4)
        abbox[i].xybox = scaledpos
        dircode = ''
        if dircodes[i][0] == 'f':
            if np.sin(cat_thetas[i]) > -buff:
                dircode += 'f'  # Face forwards
            else:
                dircode += 'b'  # Face backwards
        elif dircodes[i][0] == 'b':
            if np.sin(cat_thetas[i]) < buff:
                dircode += 'b'  # Face backwards
            else:
                dircode += 'f'  # Face forwards

        if dircodes[i][1] == 'r':
            if np.cos(cat_thetas[i]) > -buff:
                dircode += 'r'  # Face right
            else:
                dircode += 'l'  # Face left
        elif dircodes[i][1] == 'l':
            if np.cos(cat_thetas[i]) < buff:
                dircode += 'l'  # Face left
            else:
                dircode += 'r'  # Face right
        dircodes[i] = dircode
        # if np.sin(cat_thetas[i]) > 0:
        #     dircode += 'f'  # Face forwards
        # else:
        #     dircode += 'b'  # Face backward
        # if np.cos(cat_thetas[i]) > 0:
        #     dircode += 'r'  # Face right
        # else:
        #     dircode += 'l'  # Face left
        abbox[i].offsetbox = imcat[dircode][i]

    # Update labels:
    for i in range(ncat):
        shift = np.array([0, .1])
        scaledpos = scalefactor * (1.4 * (cat_pos[i] + shift) - 1.4)
        labels[i].set_position(scaledpos)

    # Update fish:
    for i in range(n):
        fishbox[i].xybox = pos[i] * scalefactor
        theta360 = (thetas[i] + np.pi) * (180. / np.pi)
        angleid = (int(theta360 + 15) % 360) / 30
        if time.time() - starttime <= t1:  # rotate
            fishbox[i].offsetbox = fishcolors[fishcolorids[i]][angleid]
        elif time.time() - starttime <= t2:
            fishbox[i].offsetbox = imcheese[angleid]
        else:
            if i in range(3):
                fishbox[i].offsetbox = imcomet[angleid]
            else:
                fishbox[i].offsetbox = imstar[angleid]

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


def data_gen_random():
    while True:
        yield np.random.uniform(0, 1, 2)


def data_gen():
    while True:
        if time.time() - starttime > 180:
            cat_pos[:] = .5
            pos[:] = .5
        elif time.time() - starttime > t0:
            pos[:], thetas[:] = cat_dynamics.timestep(
                pos, thetas, rcut, eta, vel, L, mod, cat_pos, cat_pull)
            cat_pos[:], cat_thetas[:] = cat_dynamics.timestep(
                cat_pos, cat_thetas, rcut, cat_eta, cat_vel, L)
        yield pos, thetas, cat_pos, cat_thetas


if __name__ == '__main__':
    # Define simulation parmeters and initialize dynamics:
    names = """na-young alois cory jason curtis sarah andrew soo-yeon mitch
            """.split()
    n = 20                  # Number of fish
    ncat = len(names)       # Number of cats
    L = 2.5                 # Box length
    eta = 1.2               # Fish noise term
    cat_eta = 1.            # Cat noise term
    vel = 0.1               # Fish velocity (overdamped, so constant)
    cat_vel = 0.02          # Cat velocity (overdamped, so constant)
    cat_pull = .2           # Attraction of fish to cats
    rcscale = .2            # Scale factor determining rcut
    mod = False             # Unused mod from Kranthi class
    rcut = rcscale * L      # Cutoff radius
    t0 = 1                  # Initial frozen frame to get oriented
    t1 = 10                 # Ocean
    t2 = 20                 # Field and mac n cheese
    t3 = 180                # Outer space
    t4 = 190                # End

    # Initialize:
    pos, thetas, L = cat_dynamics.initialize(n, L**2 / n)
    cat_pos = np.random.uniform(0, L, size=(ncat, 2))
    cat_thetas = np.random.uniform(-np.pi, np.pi, size=ncat)
    starttime = time.time()
    dircodes = ['fl' for _ in range(ncat)]

    # Initialize matplotlib figure
    fig, ax = plt.subplots(facecolor='black', figsize=(8, 8))
    bg1 = img.imread('f/bgocean.jpg')
    bg2 = img.imread('f/bg2.png')
    bg3 = img.imread('f/bgspace.jpg')
    bg4 = img.imread('f/endpage.png')
    im = ax.imshow(bg1, alpha=1.0, zorder=0)

    # Add cats:
    imcat = {}
    for l in ['fr', 'br', 'fl', 'bl']:
        imcat[l] = [OffsetImage(
            img.imread('icons/cats/{}{}.png'.format(i, l)), zoom=1.3)
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

    # Add fish:
    angles = range(0, 360, 30)
    ima = [OffsetImage(img.imread('icons/fish/a{}.png'.format(a)), zoom=1.3)
           for a in angles]
    imb = [OffsetImage(img.imread('icons/fish/b{}.png'.format(a)), zoom=1.3)
           for a in angles]
    imc = [OffsetImage(img.imread('icons/fish/c{}.png'.format(a)), zoom=1.3)
           for a in angles]
    imcheese = [OffsetImage(img.imread('icons/cheese/cheese{}.png'.format(a)),
                            zoom=1.) for a in angles]
    imstar = [OffsetImage(img.imread('icons/space/star{}.png'.format(a)),
                          zoom=.8) for a in angles]
    imcomet = [OffsetImage(img.imread('icons/space/comet{}.png'.format(a)),
                           zoom=1.) for a in angles]

    fishcolors = [ima, imb, imc]
    fishcolorids = np.random.randint(3, size=n)
    fishbox = [AnnotationBbox(ima[0], [0, 0], xycoords='data',
                              frameon=False) for _ in range(n)]
    for ab in fishbox:
        ab.set_zorder(1)
        ax.add_artist(ab)

    scalefactor = ax.get_xlim()[1] / L

    # Specify animation settings
    ani = animation.FuncAnimation(
        fig, update, data_gen, blit=False, interval=50)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax.axis('off')  # Turn off ugly border
    plt.show()
