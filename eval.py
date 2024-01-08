from rnn import RNN, generate_task
from rl import bo_rl
import glob

simul_ids = [int(p.split('_')[-1]) for p in glob.glob('weights/*')]
print(f'{len(simul_ids)} simul_ids found')

multipliers = [-1,
               1./4, 1./3, 1./2, 1, 2, 3, 4,
               1./4, 1./3, 1./2, 1, 2, 3, 4,
               1./4, 1./3, 1./2, 1, 2, 3, 4,
               ]
w_to_change = ['',
               'W_rec', 'W_rec', 'W_rec', 'W_rec', 'W_rec', 'W_rec', 'W_rec',
               'W_out', 'W_out','W_out', 'W_out', 'W_out','W_out', 'W_out',
               'W_in', 'W_in', 'W_in', 'W_in', 'W_in','W_in', 'W_in'
               ]

outlist = []
for i_simul in range(len(simul_ids)):
    task = generate_task(n_parallel=1, num_steps=1000)
    self = RNN(entropy_level=-1, noise_level=0.5, simul_id=simul_ids[i_simul])

    for (w, multiplier) in zip(w_to_change, multipliers):
        if multiplier != -1:
            self.scale_W(w, multiplier)
        else:
            self.load()
        cum_r, r, a = self.play(task, train=False)
        mlh, param_mlh = bo_rl(a.squeeze(), r.squeeze(), nb_batch=3, nb_BO_iter=40, verbose=True)
        alpha, beta, zeta = param_mlh[0][0], param_mlh[1][0], param_mlh[2][0]
        print(f'Case - W={w}, multiplier={multiplier}: reward is {cum_r}, mlh is {mlh}, '
              f'alpha is {alpha}, beta is {beta}, zeta is {zeta}')

        outlist.append(
            [simul_ids[i_simul], w, multiplier, cum_r[0], float(mlh), float(alpha), float(beta), float(zeta), a.squeeze().tolist(), r.squeeze().tolist()]
        )

import pandas as pd
outdf = pd.DataFrame(outlist, columns=['simul_id', "w", 'multiplier', "cum_r", "maxlkd", "alpha", "beta", "zeta", "a", "r"])

#outdf.to_parquet(f'results/res_N{outdf.simul_id.unique().size}.parquet')

# study of the results
import pandas as pd
import numpy as np
outdf = pd.read_parquet('results/res_N18.parquet')

# recover training rewards
from tensorboard.backend.event_processing import event_file_loader
from tqdm import tqdm
uniq_simul_ids_rnn = outdf.simul_id.unique()
rewards = np.zeros([uniq_simul_ids_rnn.size, 100000])

event_files = glob.glob("runs/noise_level_0_5_entropy_level_-1_simul_id_*/events.out.tfevents.*")
for event_file in tqdm(event_files):
    id_ = int(event_file.split('/')[1].split('_')[-1])
    if id_ not in uniq_simul_ids_rnn:
        continue
    loader = event_file_loader.EventFileLoader(event_file)
    idx = list(uniq_simul_ids_rnn).index(id_)
    for event in loader.Load():
        if len(event.summary.ListFields()) > 0 and event.summary.value[0].tag == 'Metrics/Reward':
            rewards[idx, event.step] = event.summary.value[-1].tensor.float_val[0]


from matplotlib import pyplot as plt
import seaborn as sns

outdf['pRepeat'] = outdf.a.apply(lambda x: np.mean(np.array(x)[1:] == np.array(x)[:-1]))
outdf['average_reward'] = outdf.cum_r / 1000
plt.figure(figsize=(16, 9))
plt.subplot(2, 3, 1)
plt.plot(rewards.T, alpha=0.1)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.ylabel('average reward during training')
plt.xlabel('training iteration')
for i, label in enumerate(["average_reward", "pRepeat", "alpha", "beta", "zeta"]):
    plt.subplot(2, 3, i + 2)
    sns.pointplot(data=outdf[outdf.w != ''], y=label, x='multiplier', hue='w', errorbar='se')
    if i >= 1:
        plt.legend([], [], frameon=False)
    else:
        plt.legend(title='')
    mean_, sem_ = outdf[outdf.w == ''][label].agg(['mean', 'sem'])
    xlim = plt.gca().get_xlim()
    plt.plot(xlim, [mean_, mean_], color='grey')
    plt.fill_between(xlim, mean_ - sem_, mean_ + sem_, alpha=0.3, color='grey')
    plt.gca().spines[['right', 'top']].set_visible(False)
plt.suptitle(f"N = {(outdf.w == '').sum()} RNN agents")
plt.tight_layout()
plt.savefig('results/out.pdf')
plt.show()
