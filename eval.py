import torch

from rnn import RNN, generate_task
import sys
from rl import fit_rl

try:
    simul_id = int(sys.argv[1]) - 1
except:
    simul_id = 3

task = generate_task(n_parallel=1, num_steps=1000)

self = RNN(entropy_level=-1, noise_level=0.5, simul_id=simul_id)
self.load()

# baseline with no gain
cum_r, r, a = self.play(task, train=False)
fit_rl(a.squeeze(), r.squeeze())
print(f'reward is {cum_r}')

# adding gain on W_rec
multiplier = 2
print()
self.scale_W('W_rec', 2)
cum_r, r, a = self.play(task, train=False)
print(f'reward is {cum_r}')
fit_rl(a.squeeze(), r.squeeze())

