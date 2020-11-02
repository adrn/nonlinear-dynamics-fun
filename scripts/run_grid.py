# Third-party
import astropy.table as at
import astropy.units as u
import numpy as np
from scipy.optimize import root
from schwimmbad.utils import batch_tasks

# gala
import gala.dynamics as gd
import gala.potential as gp
import gala.integrate as gi
from gala.units import galactic

from helpers import get_freqs


def minfunc(vy, H, E0, x, z):
    return 0.5*vy**2 + H.potential([x, 0, z]).to_value((u.km/u.s)**2) - E0


def worker(task):
    (i, j), xzvy, H = task

    norbits = xzvy.shape[0]
    freqs = np.full((3, norbits, 2), np.nan)

    for n in range(norbits):

        w0 = gd.PhaseSpacePosition(
            pos=[xzvy[n, 0], 0, xzvy[n, 1]] * u.kpc,
            vel=[0, xzvy[n, 2], 0] * u.km/u.s)

        orbit = H.integrate_orbit(w0, dt=1., t1=0, t2=256 * 300.*u.Myr,
                                  Integrator=gi.DOPRI853Integrator)

        try:
            freqs[:, n, 0] = get_freqs(orbit[:orbit.ntimes // 2])
            freqs[:, n, 1] = get_freqs(orbit[orbit.ntimes // 2:])
        except Exception:
            continue

    return (i, j), freqs


def main(pool):
    pot = gp.CCompositePotential()
    pot['disk'] = gp.MiyamotoNagaiPotential(
        m=6.5e10, a=3.5, b=0.28, units=galactic)

    pot['halo'] = gp.NFWPotential.from_M200_c(
        M200=1e12*u.Msun, c=15, units=galactic)
    pot['halo'] = gp.NFWPotential(
        m=pot['halo'].parameters['m'],
        r_s=pot['halo'].parameters['r_s'],
        c=0.9, units=galactic)

    H = gp.Hamiltonian(pot)

    # Compute a grid of orbits in x-z plane; vx=vz=0, set vy to keep E constant
    x0 = 15. * u.kpc
    vy0 = 215 * u.km/u.s
    fiducial_w0 = gd.PhaseSpacePosition(
        pos=[1., 0, 0] * x0,
        vel=[0, 1, 0] * vy0)
    E0 = H.energy(fiducial_w0)[0].to((u.km/u.s)**2)

    # Compute vy value at all grid points
    _xgrid = np.arange(5, 25+1e-3, 0.1)
    _zgrid = np.arange(0, 10+1e-3, 0.1)
    xgrid, zgrid = map(np.ravel, np.meshgrid(_xgrid, _zgrid))

    vygrid = np.full_like(xgrid, np.nan)
    for n in range(len(xgrid)):
        res = root(minfunc, 200., args=(H, E0.value, xgrid[n], zgrid[n]))
        if res.success:
            vygrid[n] = res.x

    xzvy = np.stack((xgrid, zgrid, vygrid), axis=1)

    # Create batched tasks to send out to MPI workers
    tasks = batch_tasks(n_batches=pool.size-1, arr=xzvy, args=(H, ))

    all_freqs = np.full((3, len(xgrid), 2), np.nan)
    for r in pool.map(worker, tasks):
        (i, j), freqs = r
        all_freqs[:, i:j] = freqs

    results = at.QTable()
    results['x'] = xzvy[:, 0] * u.kpc
    results['z'] = xzvy[:, 1] * u.kpc
    results['vy'] = xzvy[:, 2] * u.km/u.s
    results['freq1'] = all_freqs[..., 0].T * 1/u.Myr
    results['freq2'] = all_freqs[..., 1].T * 1/u.Myr

    results.write('orbit-freqs.fits', overwrite=True)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import sys

    # Define parser object
    parser = ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    group.add_argument("--mpiasync", dest="mpiasync", default=False,
                       action="store_true", help="Run with MPI async.")

    args = parser.parse_args()

    # deal with multiproc:
    if args.mpi:
        from schwimmbad.mpi import MPIPool
        Pool = MPIPool
        kw = dict()
    elif args.mpiasync:
        from schwimmbad.mpi import MPIAsyncPool
        Pool = MPIAsyncPool
        kw = dict()
    elif args.n_procs > 1:
        from schwimmbad import MultiPool
        Pool = MultiPool
        kw = dict(processes=args.n_procs)
    else:
        from schwimmbad import SerialPool
        Pool = SerialPool
        kw = dict()

    with Pool(**kw) as pool:
        main(pool=pool)

    sys.exit(0)
