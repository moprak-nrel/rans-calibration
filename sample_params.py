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
n_samples = 2
cmus = np.linspace(0.5 * cmu, 1.5 * cmu, n_samples)
c1s = np.linspace(0.5 * c1, 1.5 * c1, n_samples)
c2s = np.linspace(0.5 * c2, 1.5 * c2, n_samples)
params_cmu = [[cmu_, c1, c2] for cmu_ in cmus]
params_c1 = [[cmu, c1_, c2] for c1_ in c1s]
params_c2 = [[cmu, c1, c2_] for c2_ in c2s]


def run_rans_samples(Uinit, kinit, param_array, rvec):
    qois = []
    for i, params in enumerate(param_array):
        try:
            print(f"Running RANS with parameters: {params}")
            res = utils.run_rans(Uinit, kinit, params, rvec)
            qois.append(res["wake_stats"])
        except Exception:
            print(f"Failed for parameters: {params}")
            qois.append(None)
    return qois


qoi_samples = {}
qoi_samples["cmu"] = run_rans_samples(Uinit, kinit, params_cmu, rvec)
qoi_samples["c1"] = run_rans_samples(Uinit, kinit, params_c1, rvec)
qoi_samples["c2"] = run_rans_samples(Uinit, kinit, params_c2, rvec)

labels = {}
labels["cmu"] = [r"$C_\mu$" + f" = {cmu_:.3f}" for cmu_ in cmus]
labels["c1"] = [r"$C_1$" + f" = {c1_:.3f}" for c1_ in c1s]
labels["c2"] = [r"$C_2$" + f" = {c2_:.3f}" for c2_ in c2s]


def plot_qoi_variation(qois, labels, fig_name):
    color_list = plt.cm.viridis(np.linspace(0, 1, len(qois)))
    with PdfPages(fig_name) as pdf:
        plt.figure()
        for i, qoi in enumerate(qois):
            c = color_list[i]
            wake_stats = qoi
            if wake_stats is None:
                continue
            plt.plot(
                wake_stats[:, 0] / diameter,
                wake_stats[:, 1] * 2 * np.pi,
                label=labels[i],
                c=c,
                ls="solid",
            )
        plt.plot(
            (wake_stats_les[:, 0] - 1200) / diameter,
            wake_stats_les[:, 1],
            label="LES",
            color="k",
            alpha=0.5,
            ls="dotted",
        )
        plt.legend()
        plt.xlabel("x/D")
        plt.ylabel(r"Displacement Thickness ($m^2$)")
        pdf.savefig()

        plt.figure()
        for i, qoi in enumerate(qois):
            c = color_list[i]
            wake_stats = qoi
            if wake_stats is None:
                continue
            plt.plot(
                wake_stats[:, 0] / diameter,
                wake_stats[:, 2] * 2 * np.pi,
                label=labels[i],
                c=c,
                ls="solid",
            )
        plt.plot(
            (wake_stats_les[:, 0] - 1200) / diameter,
            wake_stats_les[:, 2],
            label="LES",
            color="k",
            alpha=0.5,
            ls="dotted",
        )
        plt.legend()
        plt.xlabel("x/D")
        plt.ylabel(r"Momentum Thickness ($m^2$)")
        pdf.savefig()


for p in qoi_samples.keys():
    plot_qoi_variation(qoi_samples[p], labels[p], f"{p}_samples.pdf")
