import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import utils

plt.style.use("project.mplstyle")

# Read profile data from postprocessed data
dpath = "/Users/pmohan/flowmas/SANDwake/RANSCalibration/Data"
rp = "MedWSLowTI"

wake_stats_les, vdata = utils.read_les_data(dpath, rp)
vdata = vdata[::2]  # Downsample so dr = 2 instead of dr = 1

# Set the radial grid (0, 5*R)
N = len(vdata) * 5
ri = vdata[:, 0]
rvec = np.linspace(ri[0], 5 * ri[-1], N)
diameter = 240


U = vdata[:, 1]
U_smooth = utils.smooth_u(U, N, 20, 10)
k = vdata[:, 4]
k_smooth = utils.smooth_k(k, N, 50)

# Plot the smoothed velocities as a sanity check
fig_name = "initial_conditions.pdf"
with PdfPages(fig_name) as pdf:
    plt.figure()
    plt.plot(U, ri, label="LES")
    plt.plot(U_smooth, rvec, label="Smoothed")
    plt.xlabel("U")
    plt.ylabel("R")
    plt.legend()
    pdf.savefig()
    plt.figure()
    plt.plot(k, ri, label="LES")
    plt.plot(k_smooth, rvec, label="Smoothed")
    plt.xlabel("TKE")
    plt.ylabel("R")
    plt.legend()
    pdf.savefig()

# Setup initial conditions
Uinit = U_smooth.copy()
Uinf = Uinit[-1]
kinit = k_smooth.copy()


# Setup parameter samples
cmu = 5.48 ** (-2)
c1 = 1.176
c2 = 1.92
n_samples = 5

## Run RANS with individual variations of the 3 coefficients
cmus = np.linspace(0.5 * cmu, 1.5 * cmu, n_samples)
c1s = np.linspace(0.5 * c1, 1.5 * c1, n_samples)
c2s = np.linspace(0.5 * c2, 1.5 * c2, n_samples)
params_cmu = [[cmu_, c1, c2] for cmu_ in cmus]
params_c1 = [[cmu, c1_, c2] for c1_ in c1s]
params_c2 = [[cmu, c1, c2_] for c2_ in c2s]


qoi_samples = {}
qoi_samples["cmu"] = utils.run_rans_samples(Uinit, kinit, params_cmu, rvec)
qoi_samples["c1"] = utils.run_rans_samples(Uinit, kinit, params_c1, rvec)
qoi_samples["c2"] = utils.run_rans_samples(Uinit, kinit, params_c2, rvec)

labels = {}
labels["cmu"] = [r"$C_\mu$" + f" = {cmu_:.3f}" for cmu_ in cmus]
labels["c1"] = [r"$C_1$" + f" = {c1_:.3f}" for c1_ in c1s]
labels["c2"] = [r"$C_2$" + f" = {c2_:.3f}" for c2_ in c2s]


for p in qoi_samples.keys():
    utils.plot_qoi_variation(
        qoi_samples[p], labels[p], f"{p}_samples.pdf", wake_stats_les
    )

## Run RANS simultaneously varying all the coefficients
params_list = [[cmu_, c1_, c2_] for cmu_ in cmus for c1_ in c1s for c2_ in c2s]
labels = [
    f"Cmu = {cmu_:.3f}, c1 = {c1_:.3f}, c2 = {c2_:.3f}"
    for cmu_, c1_, c2_ in params_list
]

qois = utils.run_rans_samples(Uinit, kinit, params_list, rvec)
utils.plot_qoi_variation(qois, labels, "params_samples.pdf", wake_stats_les)
