#!/usr/bin/env python
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
n = 300
ncat = 7
rho = 4.
L = np.sqrt(n / rho)
eta = 1.2
cat_eta = 1.4
vel = 0.05
cat_vel = 0.02
cat_pull = -0.3
rcscale = .05
nframes = 4000
nlog = 10
pos, thetas, L = cat_dynamics.initialize(n, rho)
cat_pos = np.random.uniform(L * .1, L * .9, size=(ncat, 2))
cat_thetas = np.random.uniform(-np.pi, np.pi, size=ncat)
rcut = rcscale * L

# Begin iterating dynamics calculations:
# for i in range(nframes):
#     if i % nlog == 0:
#         ptraj[i / nlog] = pos
#         ttraj[i / nlog] = thetas
#         cat_ptraj[i / nlog] = cat_pos
#         cat_ttraj[i / nlog] = cat_thetas
#     pos[:], thetas[:] = timestep(pos, thetas, rcut, eta, vel, L, mod,
#                                  cat_pos, cat_pull)
#     cat_pos[:], cat_thetas[:] = timestep(cat_pos, cat_thetas, rcut,
#                                          cat_eta, cat_vel, L, mod)


def data_gen():
    while True:
        for _ in range(nlog):
            cat_pos[:], cat_thetas[:] = cat_dynamics.timestep(
                cat_pos, cat_thetas, rcut, cat_eta, cat_vel, L)
        yield cat_pos, cat_thetas


# Initialize matplotlib figure
fig, ax = plt.subplots(facecolor='black', figsize=(8, 8))
# fig = plt.figure(figsize=(8,8), facecolor='black')
# ax = fig.add_subplot(111)
background = img.imread('f/bg2.png')
# background = background[::-1, :, :]  # Invert image (else it's upside down)
ax.imshow(background, alpha=0.9, zorder=0)

ibb = [OffsetImage(img.imread('icons/{}b.png'.format(i)), zoom=1.3) for
       i in range(ncat)]
ibf = [OffsetImage(img.imread('icons/{}f.png'.format(i)), zoom=1.3) for
       i in range(ncat)]
abbox = [AnnotationBbox(ib, [0, 0], xycoords='data', frameon=False) for
         ib in ibb]
for ab in abbox:
    ax.add_artist(ab)

scalefactor = ax.get_xlim()[1]
# print scalefactor, L
# imb = img.imread('icons/0b.png')
# imageboxb = OffsetImage(imb, zoom=0.8)
# imf = img.imread('icons/0f.png')
# imageboxf = OffsetImage(imf, zoom=0.8)
# # imagebox.image.axes = ax
# # Initialize in backwards facing in middle:
# cat_pos = [.5, .5]
# ab = AnnotationBbox(imageboxb, cat_pos, xycoords='data',
#                     frameon=False)
# # Then reset to forwards facing in corner:
# ab.offsetbox = imageboxf
# ab.xybox = [.9, .9]
# ax.add_artist(ab)

# Specify animation settings:


def update(data):
    cat_pos, cat_thetas = data
    for i in range(len(abbox)):
        abbox[i].xybox = cat_pos[i] * scalefactor / L
        if np.sin(cat_thetas[i]) > 0:
            abbox[i].offsetbox = ibf[i]  # Face forwards
        else:
            abbox[i].offsetbox = ibb[i]  # Face backwards
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
