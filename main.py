#%% imports

import numpy as np
import torch
import matplotlib.pyplot as plt
from model import LeakyValueRNN
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

#%% train model

epochs = 100
batch_size = 50
lmbda = 0.0 # for TD(λ)
lr = 0.003

dataloader = make_dataloader(E, batch_size=batch_size)
scores, other_scores, weights = train_model(model, dataloader, optimizer=None, epochs=epochs, lmbda=lmbda, lr=lr)
plt.figure(figsize=(3,3), dpi=300), plt.plot(scores), plt.xlabel('# epochs'), plt.ylabel('loss')

#%% probe model

E.make_trials() # create new trials for testing
dataloader = make_dataloader(E, batch_size=batch_size)
trials = probe_model(model, dataloader)[1:] # ignore first trial

#%% visualize value and rpe of example trial

plt.figure(figsize=(4,6), dpi=300)
for cue in range(E.ncues):
    clr = None
    for block in range(E.nblocks):
        linestyle = '.-' if block == 0 else '.--'
        trial = next(trial for trial in trials if trial.cue == cue and trial.block_index == block and trial.rel_trial_index >= 2)
        ts = np.arange(len(trial)) - trial.iti + 1
        plt.subplot(2,1,1)
        h = plt.plot(ts, trial.value, linestyle, label='block {}, cue {}'.format(block, cue), color=clr)
        if clr is None:
            clr = h[0].get_color()
        plt.xlim([-2, max(ts)])
        plt.ylabel('value')
        plt.legend(fontsize=8)
        plt.subplot(2,1,2)
        plt.plot(ts[:-1], trial.rpe, linestyle, color=clr)
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
plt.xticks(np.arange(Z.shape[0]), np.arange(Z.shape[0]) - E.iti_min)
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

#%%

def dprime_vector(Z1, Z2):
    """
    Compute per-feature d' between Z1 and Z2.
    
    Parameters
    ----------
    Z1 : ndarray of shape (T1, N)
        First set of samples (T1 trials, N features).
    Z2 : ndarray of shape (T2, N)
        Second set of samples (T2 trials, N features).
    
    Returns
    -------
    dprimes : ndarray of shape (N,)
        d' value for each feature.
    """
    m1 = Z1.mean(axis=0)
    m2 = Z2.mean(axis=0)
    v1 = Z1.var(axis=0, ddof=1)  # unbiased variance
    v2 = Z2.var(axis=0, ddof=1)
    
    # pooled std
    s = np.sqrt(0.5 * (v1 + v2))
    
    # avoid division by zero
    s[s == 0] = np.nan  # set zero std to NaN to avoid division by zero
    dprimes = (m1 - m2) / s
    dprimes[s == 0] = np.nan
    
    return dprimes

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

#%%
