import pathlib

# Third-party
from astropy.utils import iers
iers.conf.auto_download = False
import astropy.table as at
import astropy.units as u
import numpy as np
from scipy.optimize import root
from schwimmbad.utils import batch_tasks

# gala
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
import gala.integrate as gi
from gala.units import galactic
import superfreq as sf


def get_freqs(orbit):
    w_pp = gc.poincarepolar.cartesian_to_poincare_polar(orbit.w().T)
    fs_pp = [w_pp[:, i] + 1j*w_pp[:, i+3] for i in range(2)]

    sfreq = sf.SuperFreq(orbit.t.to_value(u.Gyr))
    res = sfreq.find_fundamental_frequencies(fs_pp)
    
    return res.fund_freqs


def worker(task):
    (i, j), w0_w, pot, Om_p, plot_path = task

    print(f"Worker {i}-{j}: running {j-i} tasks now")
    
    frame = gp.ConstantRotatingFrame(
        [0, 0, -1] * Om_p.to(u.rad/u.Myr, u.dimensionless_angles()), 
        units=galactic)
    H = gp.Hamiltonian(pot, frame)
    
    w0s = gd.PhaseSpacePosition.from_w(w0_w.T, galactic)
    static_frame = gp.StaticFrame(galactic)

    norbits = w0s.shape[0]
    freqs = np.full((2, norbits, 2), np.nan)

    for n in range(norbits):
        try:
            orbit_rot = H.integrate_orbit(
                w0s[n], dt=1., t1=0, t2=256 * 300.*u.Myr,
                Integrator=gi.DOPRI853Integrator)
            
        except RuntimeError as e:
            print(f"Worker {i}-{j}: orbit {n} failed {str(e)}")
            continue
            
        orbit = orbit_rot.to_frame(static_frame)
        split = orbit.ntimes // 2
        
        try:
            freqs[:, n, 0] = get_freqs(orbit[:split])
            freqs[:, n, 1] = get_freqs(orbit[split:])
        except Exception as e:
            print(f"Worker {i}-{j}: orbit {n} - frequencies failed {str(e)}")
            continue

    return (i, j), freqs


def main(pool):
    # make potential and hamiltonian
    mw = gp.MilkyWayPotential()

    pot = gp.CCompositePotential()
    pot['disk'] = mw['disk']
    pot['halo'] = mw['halo']
    pot['bar'] = gp.LongMuraliBarPotential(5e9, a=2., b=0.75, c=0.25, 
                                           units=galactic)
    
    Om_p = 40*u.km/u.s/u.kpc

    _R_grid = np.linspace(4, 15, 512)
    _vR_grid = np.linspace(-50, 50, 512)
    R_grid, vR_grid = list(map(np.ravel, np.meshgrid(_R_grid, _vR_grid)))

    xyz_grid = np.array([[1, 0, 0]]).T * R_grid[None] * u.kpc

    _vphi = pot.circular_velocity(xyz_grid)
    vxyz_grid = np.stack((vR_grid,
                          -_vphi.to_value(u.km/u.s),
                          np.zeros(len(_vphi)))) * u.km/u.s
    
    w0_grid = gd.PhaseSpacePosition(
        pos=xyz_grid,
        vel=vxyz_grid)

    w0_w = w0_grid.w(galactic).T
    
    plot_path = pathlib.Path('../plots/bar').resolve()
    plot_path.mkdir(exist_ok=True, parents=True)

    # Create batched tasks to send out to MPI workers
    tasks = batch_tasks(n_batches=max(pool.size-1, 1), 
                        arr=w0_w,
                        args=(pot, Om_p, plot_path))

    all_freqs = np.full((2, len(w0_w), 2), np.nan)
    for r in pool.map(worker, tasks):
        (i, j), freqs = r
        all_freqs[:, i:j] = freqs

    results = at.QTable()
    results['R'] = R_grid * u.kpc
    results['vR'] = vR_grid * u.km/u.s
    results['xyz'] = xyz_grid.T
    results['vxyz'] = vxyz_grid.T
    results['freq1'] = all_freqs[..., 0].T * u.rad/u.Gyr
    results['freq2'] = all_freqs[..., 1].T * u.rad/u.Gyr
    results.write('bar-orbit-freqs.fits', overwrite=True)


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
