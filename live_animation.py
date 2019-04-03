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
import sys
reload(sys)
sys.setdefaultencoding('utf8')

np.random.seed(0)
plt.rcParams['toolbar'] = 'None'


def plot_arrows(outdir, ptraj, cat_ptraj, ttraj, start=0,
                stop=None, stride=1, dpi=100):
    L = np.sqrt(n / rho)
    if stop is None:
        stop = len(ptraj)
    C = np.random.random(ptraj.shape[1])
    for count, frameid in enumerate(range(start, stop, stride)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        X = ptraj[frameid][:, 0]
        Y = ptraj[frameid][:, 1]
        U = np.cos(ttraj[frameid])
        V = np.sin(ttraj[frameid])
        Q = plt.quiver(X, Y, U, V, C, units='width', cmap='jet', pivot='mid')
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.scatter(*cat_ptraj[frameid].T, s=100)
        fig.savefig('{}/{}.png'.format(outdir, count),
                    dpi=100, bbox_inches='tight')
        plt.close(fig)


def initialize_fig(outdir, ptraj, cat_ptraj, ttraj, cat_ttraj, L,
                   start=0, stop=None, stride=1, dpi=100):
    ptraj = ptraj.copy()
    cat_ptraj = cat_ptraj.copy()
    if stop is None:
        stop = len(ptraj)
    C = np.random.random(ptraj.shape[1])
    background = img.imread('f/bg2.png')
    background = background[::-1, :, :]  # Invert image (else it's upside down)
    for count, frameid in enumerate(range(start, stop, stride)):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.imshow(background, alpha=0.9, zorder=0)
        scalefactor = ax.get_xlim()[1] / L
#         scalefactor = 1.
        for cat_id in range(len(cat_ptraj[frameid])):
            cat_pos = cat_ptraj[frameid][cat_id] * scalefactor
            cat_pos *= 1.2  # Extra 1.2 factor means not all cats visible
            cat_theta = cat_ttraj[frameid][cat_id]
            if np.sin(cat_theta) < 0:
                pose_code = 'f'
            else:
                pose_code = 'b'
            im = img.imread('icons/{}{}.png'.format(str(cat_id), pose_code))

            imagebox = OffsetImage(im, zoom=0.8)
            imagebox.image.axes = ax
            ab = AnnotationBbox(imagebox, cat_pos, xycoords='data',
                                frameon=False)
            ax.add_artist(ab)
        # X = ptraj[frameid][:,0] * scalefactor
        # Y = ptraj[frameid][:,1] * scalefactor
        # U = np.cos(ttraj[frameid])
        # V = np.sin(ttraj[frameid])
        # Q = plt.quiver(X, Y, U, V, C, units='width', cmap='jet', pivot='mid', zorder = 1)
        ax.set_xlim(0, L * scalefactor)
        ax.set_ylim(0, L * scalefactor)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
#         ax.scatter(*cat_ptraj[frameid].T * scalefactor, s = 1000)


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
cat_pull = -0.3
rcscale = .05
nframes = 4000
nlog = 1
pos, thetas, L = cat_dynamics.initialize(n, rho)
cat_pos = np.random.uniform(L * .1, L * .9, size=(ncat, 2))
cat_thetas = np.random.uniform(-np.pi, np.pi, size=ncat)
mod = False
rcut = rcscale * L


def data_gen():
    while True:
        for _ in range(nlog):
            pos[:], thetas[:] = cat_dynamics.timestep(
                pos, thetas, rcut, eta, vel, L, mod, cat_pos, cat_pull)
            cat_pos[:], cat_thetas[:] = cat_dynamics.timestep(
                cat_pos, cat_thetas, rcut, cat_eta, cat_vel, L)
        yield pos, thetas, cat_pos, cat_thetas


# Initialize matplotlib figure
fig, ax = plt.subplots(facecolor='black', figsize=(8, 8))
background = img.imread('f/bg2.png')
# background = background[::-1, :, :]  # Invert image (else it's upside down)
ax.imshow(background, alpha=0.9, zorder=0)

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
# fishstr = r"""
#       /`·.¸
#      /¸...¸`:·
#  ¸.·´  ¸   `·.¸.·´)
# : * ):´;      ¸  {
#  `·.¸ `·  ¸.·´`·¸)
#      `´´¸.·´
# """
# fishstr = r"""😃"""
# bbox = {'fc': '0.8', 'pad': 0}
# props = {'ha': 'center', 'va': 'center', 'bbox': bbox, 'color': 'b',
#          'fontsize': 20, 'family': 'monospace'}
# fish = [ax.text(0, 0, fishstr, props) for i in range(n)]
angles = range(0, 360, 30)
ima = [OffsetImage(img.imread('icons/fish/a{}.png'.format(a)), zoom=1.3) for
       a in angles]
fishbox = [AnnotationBbox(ima[0], [0, 0], xycoords='data', frameon=False) for
           _ in range(n)]
for ab in fishbox:
    ab.set_zorder(1)
    ax.add_artist(ab)

scalefactor = ax.get_xlim()[1]

# Specify animation settings


def update(data):
    pos, thetas, cat_pos, cat_thetas = data
    # Update cats:
    for i in range(len(abbox)):
        abbox[i].xybox = cat_pos[i] * scalefactor / L
        if np.sin(cat_thetas[i]) > 0:
            abbox[i].offsetbox = ibf[i]  # Face forwards
        else:
            abbox[i].offsetbox = ibb[i]  # Face backwards

    # Update labels:
    for i in range(ncat):
        shift = np.array([0, .1])
        labels[i].set_position((cat_pos[i] + shift) * scalefactor / L)

    # Update fish:
    for i in range(n):
        fishbox[i].xybox = pos[i] * scalefactor / L
        theta360 = (thetas[i] + np.pi) * (180. / np.pi)
        angleid = (int(theta360 + 15) % 360) / 30
        print angleid
        fishbox[i].offsetbox = ima[angleid]  # Orientation

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
