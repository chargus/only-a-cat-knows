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

np.random.seed(0)
plt.rcParams['toolbar'] = 'None'

# Define simulation parmeters and initialize dynamics:
names = """alois jason curtis sarah cory andrew na-young soo-yeon mitch
        """.split()
n = 20
ncat = len(names)
rho = 4.
L = np.sqrt(n / rho)
eta = 1.2
cat_eta = 1.4
vel = 0.05
cat_vel = 0.02
cat_pull = -1.
rcscale = .05
nframes = 4000
nlog = 1
pos, thetas, L = cat_dynamics.initialize(n, rho)
cat_pos = np.random.uniform(L * .1, L * .9, size=(ncat, 2))
cat_thetas = np.random.uniform(-np.pi, np.pi, size=ncat)
mod = False
rcut = rcscale * L
starttime = time.time()
inbg2 = False
inbg3 = False


def data_gen():
    while True:
        if time.time() - starttime > 40:
            cat_pos[:] = .5
            pos[:] = .5
        elif time.time() - starttime > 10:
            for _ in range(nlog):
                pos[:], thetas[:] = cat_dynamics.timestep(
                    pos, thetas, rcut, eta, vel, L, mod, cat_pos, cat_pull)
                cat_pos[:], cat_thetas[:] = cat_dynamics.timestep(
                    cat_pos, cat_thetas, rcut, cat_eta, cat_vel, L)
        yield pos, thetas, cat_pos, cat_thetas


# Initialize matplotlib figure
fig, ax = plt.subplots(facecolor='black', figsize=(8, 8))
bg1 = img.imread('f/bgocean.jpg')
bg2 = img.imread('f/bg2.png')
bg3 = img.imread('f/bgspace.jpg')
bg4 = img.imread('f/endpage.png')

# background = background[::-1, :, :]  # Invert image (else it's upside down)
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

# Add fish:
angles = range(0, 360, 30)
ima = [OffsetImage(img.imread('icons/fish/a{}.png'.format(a)), zoom=1.3) for
       a in angles]
imb = [OffsetImage(img.imread('icons/fish/b{}.png'.format(a)), zoom=1.3) for
       a in angles]
imc = [OffsetImage(img.imread('icons/fish/c{}.png'.format(a)), zoom=1.3) for
       a in angles]

fishcolors = [ima, imb, imc]
fishcolorids = np.random.randint(3, size=n)
fishbox = [AnnotationBbox(ima[0], [0, 0],
                          xycoords='data', frameon=False) for _ in range(n)]
for ab in fishbox:
    ab.set_zorder(1)
    ax.add_artist(ab)

scalefactor = ax.get_xlim()[1] / L

# Specify animation settings


def update(data):
    pos, thetas, cat_pos, cat_thetas = data

    # Update cats:
    for i in range(len(abbox)):
        abbox[i].xybox = cat_pos[i] * scalefactor * 1.3  # not all cats visible
        if np.sin(cat_thetas[i]) > 0:
            abbox[i].offsetbox = ibf[i]  # Face forwards
        else:
            abbox[i].offsetbox = ibb[i]  # Face backwards
        # print cat_pos[i]
        # if cat_pos[i][0] > 1 or cat_pos[i][1] > 1:
        #     abbox[i].set_alpha(.5)
        # else:
        #     abbox[i].set_alpha(1)

    # Update labels:
    for i in range(ncat):
        shift = np.array([0, .1])
        labels[i].set_position((cat_pos[i] + shift) * scalefactor * 1.3)

    # Update fish:
    for i in range(n):
        fishbox[i].xybox = pos[i] * scalefactor
        theta360 = (thetas[i] + np.pi) * (180. / np.pi)
        angleid = (int(theta360 + 15) % 360) / 30
        fishbox[i].offsetbox = fishcolors[fishcolorids[i]][angleid]  # rotate

    # Update background
    # if (not inbg2) and (time.time() - starttime > 10):

    if time.time() - starttime > 20:
        im.set_data(bg2)

    # if (not inbg3) and (time.time() - starttime > 20):
    if time.time() - starttime > 30:
        im.set_data(bg3)

    if time.time() - starttime > 40:
        im.set_data(bg4)
        im.set_zorder(101)

    if time.time() - starttime > 50:
        exit()
    return abbox,


def data_gen_random():
    while True:
        yield np.random.uniform(0, 1, 2)


ani = animation.FuncAnimation(fig, update, data_gen, blit=False, interval=25)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
# plt.box(False)
plt.tight_layout()
ax.axis('off')  # Turn off ugly border
plt.show()
