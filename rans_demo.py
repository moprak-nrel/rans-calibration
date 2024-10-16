import time

import numpy as np

import SANDRANS


def tanhUprofile(r, U0, U1, rdelta, delta):
    return 0.5 * (U1 - U0) * (1.0 + np.tanh((r - rdelta) / delta)) + U0


def run_rans(Nsteps=200, fcs=True):
    # Set RANS parameters
    params = {
        "Re": 1.0 / 1.47e-5,
        "C_mu": 0.1,
        "C_1e": 1.1,
        "C_2e": 1.92,
        "sigma_k": 1.0,
        "sigma_e": 1.3,
        "laminar": False,
    }

    # Set the radial grid
    dr = 0.025
    r1 = dr
    r2 = 5
    eps = 1.0e-6
    rvec = np.arange(r1, r2 + eps, dr)
    N = len(rvec)
    dx = 0.1

    # Setup initial conditions
    kinf = 0.001
    Uinf = 1.0
    U0 = 0.5
    U1 = 1.0
    rdelta = 1.2
    delta = 0.05
    Uinit = tanhUprofile(rvec, U0, U1, rdelta, delta)
    Vinit = np.zeros(N)
    kinit = (
        SANDRANS.set_k_init(rvec, dr, Uinit, params, k_factor=0.125)
        + kinf
        + 0 * 0.0075 * np.exp(-50 * (rvec) ** 2)
    )
    einit = SANDRANS.set_e_init(rvec, dr, Uinit, kinit, params)

    einit = SANDRANS.moving_average(einit, 10)
    phi_init = {"u": Uinit, "v": Vinit, "k": kinit, "e": einit}

    # Setup boundary conditions
    UBC = [
        {"type": "neumann", "value": 0.0},  # Lower
        {"type": "dirichlet", "value": Uinf},  # Upper
    ]
    VBC = [
        {"type": "dirichlet", "value": 0.0},  # Lower
        None,  # Upper (This is hard-coded)
    ]
    kBC = [
        {"type": "neumann", "value": 0.0},  # Lower
        {"type": "dirichlet", "value": kinf},  # Upper
    ]
    eBC = [
        {"type": "neumann", "value": 0.0},  # Lower
        {"type": "dirichlet", "value": 0.0},  # Upper
    ]
    BCdict = {"u": UBC, "v": VBC, "k": kBC, "e": eBC}

    # Set the mode registry
    St = 0.3
    moderegistry = [
        {
            "n": 1,
            "omega": np.pi * St,
            "eps": 0.004,
            "alphaguess": 0.945 * 1.5 - 0.2j,
        }
    ]
    uarray, varray, karray, earray, xvec = SANDRANS.marchWakeBL(
        phi_init,
        Nsteps,
        rvec,
        params,
        BCdict,
        dx_init=dx,
        moderegistry=moderegistry,
        LSTNr=1001,
        calcFCS=fcs,
        verbose=False,
    )

    results = {"x": xvec, "u": uarray, "k": karray, "v": varray, "e": earray}
    return results


start_time = time.time()
results = run_rans(20, False)
run_time = time.time() - start_time
print()
print("Runtime (No LST): ", run_time)

start_time = time.time()
results = run_rans(20, True)
run_time = time.time() - start_time
print()
print("Runtime (with LST): ", run_time)
