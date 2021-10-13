import matplotlib.pyplot as plt
import os
import numpy as np

FIG_DIR = 'fig/'
os.makedirs(FIG_DIR, exist_ok=True)
def savefig(fig):
    fig_savedir = str(len(os.listdir(FIG_DIR)) + 1) + '.png'
    fig_savedir = os.path.join(FIG_DIR, fig_savedir)
    fig.savefig(fig_savedir)

save=True
# save=False

# %% PSNR
xtick = ['Low resolution', 'Central', 'Distributed']
results = np.array([26.2965, 22.8740, 23.4061])
results = np.diag(results)
colors = plt.rcParams['axes.prop_cycle']
xlim = 1

fig = plt.figure(figsize=(9,6))
n_E = len(results)
x = np.array(range(n_E)) + 1
x = x*1.5
n_plot = len(results)
w = 2/n_plot
start_point = -(n_plot - 1) * (w/2)
for i, data in enumerate(zip(results, colors)):
    y, color = data
    plt.bar(x, y, width = w, color = color['color'])
# plt.plot(x, results, 'o')
plt.xticks(x, xtick, fontsize=20)
plt.xlim([x[0] - xlim, x[-1] + xlim])
plt.ylim([15, np.asarray(results).max() * 1.2 ])
plt.ylabel('PSNR', fontsize=20)
plt.grid(axis='y')
if save:
    savefig(fig)

# %% SSIM
xtick = ['Low resolution', 'Central', 'Distributed']
results = np.diag([0.7801, 0.7806, 0.7882])
colors = plt.rcParams['axes.prop_cycle']
xlim = 1

fig = plt.figure(figsize=(9,6))
n_E = len(results[0])
x = np.array(range(n_E)) + 1
x = x*1.5
n_plot = len(results)
w = 2/n_plot
start_point = -(n_plot - 1) * (w/2)
for i, data in enumerate(zip(results, colors)):
    y, color = data
    plt.bar(x, y, width = w, color = color['color'])
plt.xticks(x, xtick, fontsize=20)
plt.xlim([x[0] - xlim, x[-1] + xlim])
plt.ylim([0.777, np.asarray(results).max()*1.003])
plt.ylabel('SSIM', fontsize=20)
plt.grid(axis='y')
if save:
    savefig(fig)

# %% time - iter
xtick = ['Central', 'Distributed']
results = np.diag([3.997, 1.215])
colors = plt.rcParams['axes.prop_cycle']
xlim = 1

fig = plt.figure(figsize=(9,6))
n_E = len(results[0])
x = np.array(range(n_E)) + 1
x = x*1.5
n_plot = len(results)
w = 1.8/n_plot
start_point = -(n_plot - 1) * (w/2)
for i, data in enumerate(zip(results, colors)):
    y, color = data
    plt.bar(x, y, width = w, color = color['color'])
plt.xticks(x, xtick, fontsize=20)
plt.xlim([x[0] - xlim, x[-1] + xlim])
plt.ylim([0, np.asarray(results).max() * 1.2 ])
plt.ylabel('time per iteration [sec]', fontsize=20)
plt.grid(axis='y')
if save:
    savefig(fig)

# %% time - inference
xtick = ['Central', 'Distributed']
results = np.diag([205.9635, 49.4598])
colors = plt.rcParams['axes.prop_cycle']
xlim = 1

fig = plt.figure(figsize=(9,6))
n_E = len(results[0])
x = np.array(range(n_E)) + 1
x = x*1.5
n_plot = len(results)
w = 1.8/n_plot
start_point = -(n_plot - 1) * (w/2)
for i, data in enumerate(zip(results, colors)):
    y, color = data
    plt.bar(x, y, width = w, color = color['color'])
plt.xticks(x, xtick, fontsize=20)
plt.xlim([x[0] - xlim, x[-1] + xlim])
plt.ylim([0, np.asarray(results).max() * 1.2 ])
plt.ylabel('time for inference [sec]', fontsize=20)
plt.grid(axis='y')
if save:
    savefig(fig)
