from glob import glob

import numpy as np
from scipy.signal import savgol_filter

import SANDLST
import SANDRANS


# Gather QOIs from LES
def read_les_data(path, case):
    wake_files = glob(f'{path}/{case}*thickness*')
    wake_stats_les = None
    for wf in wake_files:
        thicknessdata = np.loadtxt(wf, skiprows=1, delimiter=',')
        if wake_stats_les is None:
            wake_stats_les = thicknessdata
        else:
            wake_stats_les = np.vstack([wake_stats_les, thicknessdata])

    # Read initial profiles from LES
    vfile = f"{path}/{case}_YZwake1_rprofile_0.csv"
    vdata = np.loadtxt(vfile,skiprows=1,delimiter=',')
    return wake_stats_les, vdata

# Smooth the velocity profile
def smooth_u(v, N, lclip, order = 5):
    v_mod = np.zeros(N)
    v_mod[:len(v)] = v
    v_mod[:lclip] = v_mod[lclip]
    rclip = np.argmax(v_mod)
    v_mod[rclip:]= v_mod[rclip]
    return savgol_filter(v_mod, order, 3)

# Smooth the tke profile
def smooth_k(k, N, lclip):
    k_mod = np.zeros(N)
    k_mod[:len(k)] = k
    k_mod[:lclip] = k_mod[lclip]
    rclip = lclip + (len(k) - lclip)//2+ np.argmin(k[lclip+(len(k)-lclip)//2:])
    k_mod[rclip:] = k[rclip]
    return savgol_filter(k_mod, 5, 3)

# Setup RANS and run with the velocity/tke profiles from data
def run_rans(Uinit, kinit, calib_params, rvec, dx=10, Nsteps=250):
    # Set RANS parameters
    params = {
        "Re": 1.0 / 1.77e-5,
        "C_mu": 5.48 ** (-2),
        "C_1e": 1.176,
        "C_2e": 1.92,
        "sigma_k": 1.0,
        "sigma_e": 1.3,
        "laminar": False,
    }

    # Override with caliberation parameters
    params["C_mu"] = calib_params[0]
    params["C_1e"] = calib_params[1]
    params["C_2e"] = calib_params[2]

    # Uinf = np.max(Uinit)
    Uinf = Uinit[-1]
    Vinit = np.zeros_like(rvec)
    dr = rvec[1] - rvec[0]
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

    # %%
    # Gather QOIs: currently wake thickness, centerline velocity
    # x_outputs = np.array(xvec[::5])
    x_outputs = np.array(xvec)
    wake_stats = np.empty((len(x_outputs), 4))
    for i, x in enumerate(x_outputs):
        wake_stats[i, 0] = x
        idx = np.searchsorted(xvec, x)
        wake_stats[i, 1] = SANDLST.calcDeltaThick(
            uarrayNOFCS[idx, :], Uinf, rvec
        )  # wake displacement
        wake_stats[i, 2] = SANDLST.calcDeltaMom(
            uarrayNOFCS[idx, :], Uinf, rvec
        )  # wake momentum thickness
        wake_stats[i, 3] = np.min(uarrayNOFCS[idx, :])  # center line velocity
    results = {"wake_stats": wake_stats, "u": uarrayNOFCS, "k": karrayNOFCS}
    return results
