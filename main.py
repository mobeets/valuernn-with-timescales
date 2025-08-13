#%% imports

import numpy as np
import torch
import matplotlib.pyplot as plt
from model import LeakyValueRNN
from analysis import dprime_vector
from valuernn.train import make_dataloader, train_model, probe_model
from valuernn.tasks.inference import ValueInference

#%% make trials

E = ValueInference(nepisodes=500,
            nblocks_per_episode=4,
            ntrials_per_block=10,
            reward_times_per_block=(2,2), # time steps between cue and reward
            iti_p=1.0, # iti_p = 1 -> deterministic ITI
            first_block_is_random=False,
            reward_sizes_per_block={0: (1.0,0.1), 1: (0.1,1.0)}, # {cue: (blk1, blk2)}
            reward_probs_per_block={0: (1,1), 1: (1,1)}, # {cue: (blk1, blk2)}
            )
# note: much more success when first block is always the same; i suspect we might be able to handle a random first block is the rnn could also learn its initial state

#%% make model

hidden_size = 50 # number of hidden neurons
comm_size = 1 # number of communication units

gamma = 0.9 # discount factor
alpha_dist = torch.Tensor((hidden_size//2)*[0.1] + (hidden_size//2)*[1.0])
assert hidden_size % 2 == 0, "hidden_size must be even for alpha_dist to work"

model = LeakyValueRNN(input_size=E.ncues + E.nrewards,
                 output_size=E.nrewards,
                 alpha_dist=alpha_dist,
                 comm_size=comm_size,
                 hidden_size=hidden_size, gamma=gamma)

# plot distribution of integration timescales across units
plt.figure(figsize=(3,3), dpi=300)
if comm_size is None:
    plt.hist(model.rnn.cell.alpha, bins=np.linspace(0,1,11))
else:
    plt.hist(model.rnn.f1.cell.alpha, bins=np.linspace(0,1,11), alpha=0.8)
    plt.hist(model.rnn.f2.cell.alpha, bins=np.linspace(0,1,11), alpha=0.8)
plt.xlim([0,1])
plt.xlabel('timescale of integration')
plt.ylabel('number of units')

#%% load weights from path

fnm = 'models/leaky-rnn-model-50.pth'
model.load_weights_from_path(fnm)

#%% train model

epochs = 100
batch_size = 50
lmbda = 0.0 # for TD(λ)
lr = 0.003

dataloader = make_dataloader(E, batch_size=batch_size)
scores, other_scores, weights = train_model(model, dataloader, optimizer=None, epochs=epochs, lmbda=lmbda, lr=lr)
plt.figure(figsize=(3,3), dpi=300), plt.plot(scores), plt.xlabel('# epochs'), plt.ylabel('loss')

#%% probe model

inactivation_indices = None
# inactivation_indices = np.arange(hidden_size // 2) # silence pop 1
# inactivation_indices = np.arange(hidden_size // 2, hidden_size) # silence pop 2

E.make_trials() # create new trials for testing
dataloader = make_dataloader(E, batch_size=batch_size)
trials = probe_model(model, dataloader, inactivation_indices=inactivation_indices)[1:] # ignore first trial

#%% visualize value and rpe of example trial

showRPE = False

plt.figure(figsize=(4,6), dpi=300)
for cue in range(E.ncues):
    clr = None
    for block in range(E.nblocks):
        linestyle = '.-' if block == 0 else '.--'
        
        cur_trials = [trial for trial in trials if trial.cue == cue and trial.block_index == block and trial.rel_trial_index >= 2]

        vs = np.dstack([trial.value[trial.iti-E.iti_min:] for trial in cur_trials]).mean(axis=-1)
        rpes = np.dstack([trial.rpe[trial.iti-E.iti_min:] for trial in cur_trials]).mean(axis=-1)

        ts = np.arange(len(trial)) - trial.iti + 1
        plt.subplot(2,1,1)
        h = plt.plot(ts, vs, linestyle, label='block {}, cue {}'.format(block, cue), color=clr)
        if clr is None:
            clr = h[0].get_color()
        plt.xlim([-2, max(ts)])
        plt.ylabel('value')
        plt.legend(fontsize=8)
        if showRPE:
            plt.subplot(2,1,2)
            plt.plot(ts[:-1], rpes, linestyle, color=clr)
            plt.xlim([-2, max(ts)])
            plt.ylabel('RPE')
        plt.xlabel('time rel. to cue onset')
plt.tight_layout()

#%% visualize heatmap of unit activity alongside timescale integration

mus = {}
for block in range(E.nblocks):
    for cue in range(E.ncues):
        Zs = []
        for trial in trials:
            if trial.rel_trial_index <= 2:
                continue
            if trial.block_index == block and trial.cue == cue:
                continue
            Zs.append(trial.Z[trial.iti-E.iti_min:])
        Z = np.dstack(Zs).mean(axis=-1)
        mus[(block, cue)] = Z.copy()

Z = (Z - Z.mean(axis=0)) / Z.std(axis=0) # z-score each unit
ixm = np.argmax(Z, axis=0) # find time of peak for each unit
ix = np.argsort(ixm) # sort by time of peak
# ix = np.arange(Z.shape[1]) # sort indices

plt.figure(figsize=(4,3), dpi=300)
plt.imshow(Z.T[ix], aspect='auto', cmap='viridis')
plt.xticks(np.arange(Z.shape[0]), np.arange(Z.shape[0]) - E.iti_min + 1)
plt.colorbar()

ys = np.arange(Z.shape[1])
if comm_size is None:
    alphas = model.rnn.cell.alpha.numpy()
else:
    alphas = np.hstack([model.rnn.f1.cell.alpha.numpy(), model.rnn.f2.cell.alpha.numpy()])
xs = alphas[ix] * (Z.shape[0]-1)
plt.plot(xs, ys, 'o', color='white', markersize=2, alpha=0.5)
plt.xlabel('time steps rel. to cue onset')
plt.ylabel('units (sorted by peak response time)')

#%% find block-representing and cue-representing units, and compare to alpha

mus = {'block': [], 'cue': []}
dprimes = {'block': [], 'cue': []}

# t_pre = E.iti_min
t_pre = 1
t_post = 1

for block in range(E.nblocks):
    Zs_block = []
    for trial in trials:
        if trial.rel_trial_index <= 2:
            continue
        if trial.block_index == block:
            continue
        # Zs_block.append(trial.Z[trial.iti-t_pre:trial.iti].mean(axis=0))
        Zs_block.extend(trial.Z[trial.iti-t_pre:trial.iti])
    Z = np.dstack(Zs_block).T[:,:,0]
    mus['block'].append(Z.copy())
assert len(mus['block']) == 2
dprimes['block'] = np.abs(dprime_vector(mus['block'][0], mus['block'][1]))

block_for_cue_dprime = 0
for cue in range(E.ncues):
    Zs_cue = []
    for trial in trials:
        if trial.rel_trial_index <= 2:
            continue
        if trial.block_index != block_for_cue_dprime:
            continue
        if trial.cue == cue:
            continue
        # Zs_cue.append(trial.Z[trial.iti:trial.iti+trial.isi].mean(axis=0))
        Zs_cue.extend(trial.Z[trial.iti:trial.iti+t_post])
    Z = np.dstack(Zs_cue).T[:,:,0]
    mus['cue'].append(Z.copy())
assert len(mus['cue']) == 2
dprimes['cue'] = np.abs(dprime_vector(mus['cue'][0], mus['cue'][1]))

Zss = []
Zss.append(dprimes['block'])
Zss.append(dprimes['cue'])
if comm_size is None:
    alphas = model.rnn.cell.alpha.numpy()
else:
    alphas = np.hstack([model.rnn.f1.cell.alpha.numpy(), model.rnn.f2.cell.alpha.numpy()])
Zss.append(alphas)
Z = np.vstack(Zss)
print(Z.shape)

# sort units by timescale
ix = np.argsort(Z[-1])
Z = Z[:, ix]

plt.figure(figsize=(3,3), dpi=300)
keys = list(mus.keys())
for i, z in enumerate(Z):
    if i >= len(keys):
        continue
    plt.plot(z, label=keys[i] if i < len(keys) else 'alpha')
plt.xlabel('unit index (sorted by alpha)')
plt.ylabel("|d'|")
plt.legend(fontsize=8)
plt.show()

if np.unique(alphas).size == 2:
    # plot hist of d' for each value of alpha
    plt.figure(figsize=(6,3), dpi=300)
    for alpha in np.unique(alphas):
        plt.subplot(1, 2, 1)
        bins = np.linspace(0, np.nanmax(dprimes['block']), 20)
        h = plt.hist(dprimes['block'][alphas == alpha], bins=bins, alpha=0.5, label=f"α={alpha:.2f}")
        # plot vertical line at median d'
        clr = h[-1][-1].get_facecolor()
        plt.axvline(np.nanmedian(dprimes['block'][alphas == alpha]), color=clr, linestyle='--')
        plt.xlabel("|d'| block")
        plt.ylabel("number of units")
        plt.legend(fontsize=8)

        plt.subplot(1, 2, 2)
        bins = np.linspace(0, np.nanmax(dprimes['cue']), 20)
        h = plt.hist(dprimes['cue'][alphas == alpha], bins=bins, alpha=0.5, label=f"α={alpha:.2f}")
        # plot vertical line at median d'
        clr = h[-1][-1].get_facecolor()
        plt.axvline(np.nanmedian(dprimes['cue'][alphas == alpha]), color=clr, linestyle='--')
        plt.xlabel("|d'| cue")
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(3,3), dpi=300)
row_means = np.nanmean(Z, axis=1, keepdims=True)
Z_filled = np.where(np.isnan(Z), row_means, Z)
corr_matrix = np.corrcoef(Z_filled)
plt.imshow(corr_matrix, cmap='RdBu')
plt.xticks([0,1,2], ['block |d\'|', 'cue |d\'|', 'alpha'])
plt.yticks([0,1,2], ['block |d\'|', 'cue |d\'|', 'alpha'])
# plt.axis('off')
plt.clim([-1,1])
plt.colorbar(fraction=0.03, pad=0.04)
plt.show()

#%% visualize average activity of low and high d' units

dps = dprimes['block'].copy()
dps[np.isnan(dps)] = np.nanmedian(dps) # replace NaNs with median
block_dprimes = np.argsort(dps)
group1 = block_dprimes[:10]
group2 = block_dprimes[-10:]
names = ['low d\'', 'high d\'']

# group1 = np.arange(25); group2 = np.arange(25, 50); names = ['pop 1', 'pop 2']

print(dps[group1].mean(), dps[group2].mean())

plt.figure(figsize=(6,3), dpi=300)
mus = {}
for block in range(E.nblocks):
    for cue in range(E.ncues):
        Zs = []
        for trial in trials:
            if trial.rel_trial_index <= 2:
                continue
            if trial.block_index != block:
                continue
            if trial.cue != cue:
                continue
            Zs.append(trial.Z[(trial.iti-E.iti_min):])
        Z = np.dstack(Zs).mean(axis=-1) # average across trials
        mus[(block, cue)] = Z.copy()

        xs = np.arange(Z.shape[0]) - E.iti_min + 1
        plt.subplot(1,2,1)
        plt.plot(xs, Z[:,group1].mean(axis=1), '-' if block == 0 else '--', color='blue' if cue == 0 else 'orange', label=f'cue={cue}, block={block}')
        plt.title(names[0])
        plt.subplot(1,2,2)
        plt.plot(xs, Z[:,group2].mean(axis=1), '-' if block == 0 else '--', color='blue' if cue == 0 else 'orange', label=f'cue={cue}, block={block}')
        plt.xlabel('time rel. to cue onset')
        plt.ylabel('activity')
        plt.title(names[1])
        plt.legend(fontsize=8)
plt.tight_layout()
