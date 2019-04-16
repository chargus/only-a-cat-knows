"""
Adapted from projects/03_active/active.py

"""

import numpy as np
# np.random.seed(0)


def initialize(n, rho):
    """Set up initial positions and angles in the simulation box.

    Parameters
    ----------
    n : int
        Number of particles in the simulation
    rho : float
        Density
    """
    L = np.sqrt(n / rho)
    pos = np.random.uniform(0, L, size=(n, 2))
    thetas = np.random.uniform(-np.pi, np.pi, size=n)
    return pos, thetas, L


def va(thetas):
    xs = np.cos(thetas)
    ys = np.sin(thetas)
    return np.sqrt(np.mean(xs) + np.mean(ys))


def va_traj(ttraj):
    xs = np.cos(ttraj)
    ys = np.sin(ttraj)
    return np.sqrt(np.mean(xs, axis=1)**2 + np.mean(ys, axis=1)**2)


def avg_theta(thetas, cat_vec=None, cat_pull=0.):
    """Compute average orientation.

    Note: Cannot simply take the mean of all theta values, since this
    does not correclty account for the discontinuity from theta = -pi to pi.

    """
    xmean = np.mean(np.sin(thetas))
    ymean = np.mean(np.cos(thetas))

    if cat_vec is not None:
        nn = np.argmin(np.sum(cat_vec**2, axis=1), axis=0)
        xmean = (xmean + cat_pull * cat_vec[nn][0])
        ymean = (ymean + cat_pull * cat_vec[nn][1])
    return np.arctan2(xmean, ymean)


def apply_pbc(pos, L):
    """Apply periodic boundary conditions to an array of positions.

    It is assumed that the simulation box exists on domain [-L/2, L/2]^D
    where D is the number of dimensions.

    Parameters
    ----------
    pos : numpy ndarray
        Description
    L : float
        Description

    Returns
    -------
    pos:
        Description
    """
    return ((pos + L / 2) % L) - L / 2


def get_neighbors(pos, rcut, L=1.0):
    """Find neighbors within cutoff distance, including across PBCs.
    (returns a mask)
    """
    dx = np.subtract.outer(pos[:, 0], pos[:, 0])
    dy = np.subtract.outer(pos[:, 1], pos[:, 1])

    # Apply "minimum image" convention: interact with nearest periodic image
    dx = apply_pbc(dx, L)  # x component of i-j vector
    dy = apply_pbc(dy, L)  # y component of i-j vector

    r2 = dx**2 + dy**2  # Squared distance between all particle pairs

    # Select interaction pairs within cutoff distance
    # (also ignore self-interactions)
    mask = r2 < rcut**2  # Note: include self in alignment average.
    return mask


def align(pos, thetas, eta, rcut, L=1.0, mod=False, cat_pos=None, cat_pull=0.):
    """Align all particles to the velocity vector of their neighbors.
    """
    mask = get_neighbors(pos, rcut, L)
    thetas_new = np.empty_like(thetas)
    # if cat_pos is not None:
    #     nearest_cat = np.argmin(scipy.spatial.distance.cdist(pos, cat_pos),
    #                             axis=1)
    # note: no easy way to vectorize for-loop, since len of thetas_new varies
    for i in range(len(thetas)):
        neighbors_thetas = thetas[mask[i]]
        if cat_pos is not None:
            cat_vec = cat_pos - pos[i]
        else:
            cat_vec = None
        avg = avg_theta(neighbors_thetas, cat_vec, cat_pull)
        thetas_new[i] = avg
    if mod:
        rho_global = len(mask) / L**2
        rho_local = sum(mask) / rcut**2
        eta = (eta * rho_local / rho_global)  # eta is different for each atom
    thetas_new += np.random.uniform(-eta / 2., eta / 2., len(thetas))
    return thetas_new


def timestep(pos, thetas, rcut, eta, vel, L,
             mod=False, cat_pos=None, cat_pull=0.):
    """Update position and angles of all particles.
    """
    # First update the positions:
    dx = vel * np.cos(thetas)
    dy = vel * np.sin(thetas)
    pos[:, 0] = (pos[:, 0] + dx) % L  # Modulo L to handle PBCs
    pos[:, 1] = (pos[:, 1] + dy) % L  # Modulo L to handle PBCs
    # Now update the angles:
    thetas = align(pos, thetas, eta, rcut, L, mod, cat_pos, cat_pull)
    return pos, thetas


def get_r2(pos):
    # Determine particles within cutoff radius
    dx = np.subtract.outer(pos[:, 0], pos[:, 0])
    dy = np.subtract.outer(pos[:, 1], pos[:, 1])

    # Apply "minimum image" convention: interact with nearest periodic image
    # dx = apply_pbc(dx, L)  # x component of i-j vector
    # dy = apply_pbc(dy, L)  # y component of i-j vector

    r2 = dx**2 + dy**2  # Squared distance between all particle pairs
    return r2


def lj(r2, sigma, epsilon):
    """Compute the Lennard-Jones force and potential energy.

    Returned force is normalized by r, such that it can be multiplied by the
    x, y or z component of r to obtain the correct cartesian component of the
    force.

    Parameters
    ----------
    r2 : float or [N] numpy array
        Squared particle-particle separation.
    sigma : float
        Width of the potential, defined by distance at which potential is zero.
    epsilon : float
        Depth of the potential well, relative to the energy at infinite
        separation.

    """
    r12 = (sigma**2 / r2)**6
    r6 = (sigma**2 / r2)**3
    lj_force = (48 / r2) * epsilon * (r12 - .5 * r6)
    lj_energy = 4 * epsilon * (r12 - r6)
    return lj_force, lj_energy


def rep12(r2, sigma, epsilon):
    """Compute a purely repulsive potential and corresponding force.

    Returned force is normalized by r, such that it can be multiplied by the
    x, y or z component of r to obtain the correct cartesian component of the
    force.

    Parameters
    ----------
    r2 : float or [N] numpy array
        Squared particle-particle separation.
    sigma : float
        Width of the potential, defined by distance at which potential is zero.
    epsilon : float
        Depth of the potential well, relative to the energy at infinite
        separation.

    """
    r12 = (sigma**2 / r2)**6
    force = (48 / r2) * epsilon * r12
    energy = 4 * epsilon * r12
    return force, energy


def timestep_underdamped(pos, vel, gamma, T, rcut, L, dt,
                         sigma=1., epsilon=1.):
    """Update position and angles of all particles.
    """
    # First update the positions:
    R = np.random.normal(size=(len(pos), 2)) * np.sqrt(2 * T * gamma * dt)

    dx = np.subtract.outer(pos[:, 0], pos[:, 0])
    dy = np.subtract.outer(pos[:, 1], pos[:, 1])
    # Apply "minimum image" convention: interact with nearest periodic image
    # dx = apply_pbc(dx, L)  # x component of i-j vector
    # dy = apply_pbc(dy, L)  # y component of i-j vector
    r2 = dx**2 + dy**2  # Squared distance between all particle pairs
    mask = r2 > 0
    lj_force, _ = lj(r2[mask], sigma, epsilon)
    fx = np.zeros_like(dx)
    fx[mask] = -dx[mask] * lj_force  # Negative sign so dx points from j to i
    fy = np.zeros_like(dy)
    fy[mask] = -dy[mask] * lj_force  # Negative sign so dy points from j to i
    net_fx = np.sum(fx, axis=0)
    net_fy = np.sum(fy, axis=0)
    F = np.stack([net_fx, net_fy], axis=1)

    pos[:] = (pos[:] + vel * dt) % L  # Modulo L to handle PBCs
    vel[:] = (vel[:] - gamma * vel[:] + F + R[:])
    # vel[:, 0] = (vel[:, 0] - gamma * vel[:, 0] + fx + R[:, 0])
    # vel[:, 1] = (vel[:, 1] - gamma * vel[:, 1] + fy + R[:, 1])
    # Now update the angles:
    return pos, vel


def piano_trio_bc(pos, vel, L):
    lb = 0.17 * L  # Lower bound
    ub = L  # Upper bound
    # Left
    mask = pos[:, 0] < lb
    pos[:, 0][mask] = lb                 # move in-bounds
    vel[:, 0][mask] = -vel[:, 0][mask]   # bounce velocity

    # Right
    mask = pos[:, 0] > ub
    pos[:, 0][mask] = ub                 # move in-bounds
    vel[:, 0][mask] = -vel[:, 0][mask]   # bounce velocity

    # Bottom
    mask = pos[:, 1] < lb
    pos[:, 1][mask] = lb                 # move in-bounds
    vel[:, 1][mask] = -vel[:, 1][mask]   # bounce velocity

    # Top
    mask = pos[:, 1] > ub
    pos[:, 1][mask] = ub                 # move in-bounds
    vel[:, 1][mask] = -vel[:, 1][mask]   # bounce velocity

    # Boat
    # blb = 0.5 * L
    # bub = 0.6 * L
    # bleftb = 0.4 * L
    # brightb = 0.6 * L
    # mask = (pos[:, 1] > blb) * (pos[:, 1] < bub)
    # mask *= (pos[:, 0] > bleftb) * (pos[:, 1] < brightb)
    # mask_down = mask * (vel[:, 1] >= 0)
    # mask_up = mask * (vel[:, 1] < 0)
    # pos[:, 1][mask_down] = blb
    # pos[:, 1][mask_up] = bub
    # vel[:, 1][mask] = -vel[:, 1][mask]

    return pos, vel


def timestep_newton(pos, vel, L, dt, sigma=1., epsilon=1., m=1.):
    """Update position and angles of all particles.
    """
    # First update the positions:
    x = pos[:, 0]
    y = pos[:, 1]
    dx = np.subtract.outer(x, x)
    dy = np.subtract.outer(y, y)
    r2 = dx**2 + dy**2  # Squared distance between all particle pairs
    mask = r2 > 0
    force, _ = rep12(r2[mask], sigma, epsilon)
    fx = np.zeros_like(dx)
    fx[mask] = -dx[mask] * force  # Negative sign so dx points from j to i
    fy = np.zeros_like(dy)
    fy[mask] = -dy[mask] * force  # Negative sign so dy points from j to i
    net_fx = np.sum(fx, axis=0)
    net_fy = np.sum(fy, axis=0)

    F = np.stack([net_fx, net_fy], axis=1)
    # pos[:] = (pos[:] + vel * dt) % L      # Modulo L to handle PBCs
    pos[:] = (pos[:] + vel * dt)
    vel[:] = (vel[:] + F * dt / m[:, None])
    return pos, vel


def run(n, ncat, rho, eta, cat_eta, vel, cat_vel, cat_pull, rcut=None,
        nframes=100, nlog=10, mod=False, rcscale=None):
    """Run a dynamics simulation.

    Parameters
    ----------
    n : int
        Number of particles.
    rho : float
        Density (n / L**2).
    eta : float
        Specifies range for angular noise: dtheta ~ [-eta/2, eta/2]
    rcut : float
        Factor for cutoff radius beyond which interactions are not considered.
        This is a factor of L, e.g. 0.5 gives half of L as the the cutoff.
        If None, value of 0.5 is used
    nframes : int
        Number of frames for which to run simulation.
    nlog : int
        Log positions and kinetic energy to trajectories every `nlog` frames.
    mod : bool
        If true, use local density noise modification.
    """
    pos, thetas, L = initialize(n, rho)
    cat_pos = np.random.uniform(L * .1, L * .9, size=(ncat, 2))
    cat_thetas = np.random.uniform(-np.pi, np.pi, size=ncat)
    if rcut > L / 2:
        raise ValueError("rcut must be less than or equal to L/2 to satisfy"
                         "the minimum image convention.")
    if rcut is None:
        if rcscale is None:
            rcscale = 0.5  # If not provided, use max possible rcscale
        rcut = rcscale * L

    # Initialize arrays:
    ptraj = np.empty((nframes / nlog, pos.shape[0], pos.shape[1]))
    ttraj = np.empty((nframes / nlog, pos.shape[0]))
    cat_ptraj = np.empty((nframes / nlog, cat_pos.shape[0], cat_pos.shape[1]))
    cat_ttraj = np.empty((nframes / nlog, cat_pos.shape[0]))

    # Begin iterating dynamics calculations:
    for i in range(nframes):
        if i % nlog == 0:
            ptraj[i / nlog] = pos
            ttraj[i / nlog] = thetas
            cat_ptraj[i / nlog] = cat_pos
            cat_ttraj[i / nlog] = cat_thetas
        pos[:], thetas[:] = timestep(pos, thetas, rcut, eta, vel, L, mod,
                                     cat_pos, cat_pull)
        cat_pos[:], cat_thetas[:] = timestep(cat_pos, cat_thetas, rcut,
                                             cat_eta, cat_vel, L, mod)

    return ptraj, ttraj, cat_ptraj, cat_ttraj
