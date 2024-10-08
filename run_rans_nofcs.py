import numpy as np
import SANDLST
import SANDRANS


def tanhUprofile(r, U0, U1, rdelta, delta):
    return 0.5 * (U1 - U0) * (1.0 + np.tanh((r - rdelta) / delta)) + U0


def run_rans(calib_params=[5.48 ** (-2), 1.176, 1.92]):
    # Set RANS parameters
    params = {
        "Re": 1.0 / 1.47e-5,
        "C_mu": 5.48 ** (-2),
        "C_1e": 1.176,
        "C_2e": 1.92,
        "sigma_k": 1.0,
        "sigma_e": 1.3,
        "laminar": False,
    }

    # Override with values from MCMC
    params["C_mu"] = calib_params[0]
    params["C_1e"] = calib_params[1]
    params["C_2e"] = calib_params[2]

    # Set the radial grid
    dr = 0.025
    rcl = dr
    rinf = 5
    eps = 1.0e-6
    rvec = np.arange(rcl, rinf + eps, dr)
    N = len(rvec)
    dx = 0.1
    Nsteps = 200

    # Setup initial conditions
    U0 = 0.65
    Uinf = 1.0
    rdelta = 1.0
    delta = 0.1
    Uinit = tanhUprofile(rvec, U0, Uinf, rdelta, delta)
    Vinit = np.zeros(N)
    kinit = SANDRANS.set_k_init(rvec, dr, Uinit, params, k_factor=0.15)
    einit = SANDRANS.set_e_init(rvec, dr, Uinit, kinit, params)
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
        {"type": "dirichlet", "value": 0.0},  # Upper
    ]
    eBC = [
        {"type": "neumann", "value": 0.0},  # Lower
        {"type": "dirichlet", "value": 0.0},  # Upper
    ]
    BCdict = {"u": UBC, "v": VBC, "k": kBC, "e": eBC}

    # No modes for calibration
    moderegistryNOFCS = []

    # Compute the wake using RANS
    uarrayNOFCS, varrayNOFCS, karrayNOFCS, earrayNOFCS, xvec = SANDRANS.marchWakeBL(
        phi_init,
        Nsteps,
        rvec,
        params,
        BCdict,
        dx_init=dx,
        moderegistry=moderegistryNOFCS,
        LSTNr=1001,
        calcFCS=False,
        verbose=False,
    )

    # Gather calibration outputs, currently wake thickness, centerline velocity
    x_outputs = [Nsteps // 4, 2 * Nsteps // 4, 3 * Nsteps // 4, -1]
    wake_stats = np.empty((len(x_outputs), 4))
    for ix, x in enumerate(x_outputs):
        # xloc
        wake_stats[ix, 0] = xvec[x]
        # wake displacement
        wake_stats[ix, 1] = SANDLST.calcDeltaThick(uarrayNOFCS[x, :], Uinf, rvec)
        # wake momentum thickness
        wake_stats[ix, 2] = SANDLST.calcDeltaMom(uarrayNOFCS[x, :], Uinf, rvec)
        # center line velocity
        wake_stats[ix, 3] = np.min(uarrayNOFCS[x, :])
    return wake_stats
