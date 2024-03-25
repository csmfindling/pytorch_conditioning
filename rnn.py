import numpy as np
from scipy.stats import truncnorm
import torch
from torch import nn
from torch.optim import Adam
import scipy
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def generate_task(n_parallel, num_steps):
    proba_r = np.zeros([num_steps, n_parallel]) + 0.5
    for i in range(1, num_steps):
        a, b = (0 - proba_r[i-1]) / 0.108, (1 - proba_r[i-1]) / 0.108
        proba_r[i] = truncnorm.rvs(a, b, loc=proba_r[i-1], scale=0.108)
    a, b = (0 - proba_r) / 0.153, (1 - proba_r) / 0.153
    rewards = np.zeros([num_steps, n_parallel, 2])
    rewards[:, :, 0] = truncnorm.rvs(a, b, loc=proba_r, scale=0.153)
    rewards[:, :, 1] = 1 - rewards[:, :, 0]
    return torch.from_numpy(rewards) - 0.5

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class RNN(nn.Module):
    def __init__(self, num_units=64, input_size=3, noise_level=-1, entropy_level=-1, name='', nb_max_epochs=1000, gamma=0.5, simul_id=0):
        super(RNN, self).__init__()
        self.num_units = num_units
        self.input_size = input_size
        self.with_noise = noise_level > 0
        self.with_entropy = entropy_level > 0
        self.nb_max_epochs = nb_max_epochs
        self.gamma = gamma
        self.name = 'RNN' + name + '_noisy' * self.with_noise
        self.W_rec = torch.nn.Parameter(torch.zeros([self.num_units, self.num_units]))
        self.W_in = torch.nn.Parameter(torch.zeros([self.input_size, self.num_units]))
        self.b_rec = torch.nn.Parameter(torch.zeros([self.num_units]))
        self.W_out = torch.nn.Parameter(torch.zeros([self.num_units, 2]))
        self.noise_level = noise_level
        self.entropy_level = entropy_level
        if torch.backends.mps.is_available() and False:
            print('moving to mps')
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.to(device=self.device)
        self.path_to_weights = Path('weights')
        self.path_to_weights.mkdir(exist_ok=True)
        self.model_name = "noise_level_{}_entropy_level_{}_simul_id_{}".format(
                str(self.noise_level).replace('.', '_'),
                str(self.entropy_level).replace('.', '_'),
                str(simul_id))

    def reset_weights(self):
        with torch.no_grad():
            torch.nn.init.xavier_uniform(self.W_rec)
            torch.nn.init.xavier_uniform(self.W_in)
            torch.nn.init.xavier_uniform(self.W_out)
            self.b_rec[:] = 0

    def forward(self, prev_act, prev_rew, h, n_parallel):
        input_ = torch.hstack((torch.nn.functional.one_hot(prev_act, num_classes=2), prev_rew[:, None])).to(torch.float32).to(self.device)
        h = torch.matmul(input_, self.W_in) + torch.matmul(h, self.W_rec) + self.b_rec
        if self.with_noise:
            h = h + torch.normal(mean=0., std=self.noise_level, size=(n_parallel, self.num_units,))
        h = torch.nn.Tanh()(h)
        return torch.nn.Softmax(dim=1)(torch.matmul(h, self.W_out)), h

    def play(self, rewards=None, train=True, plot_results=False):
        if not train and rewards is None:
            raise AssertionError('reward can not be null if train is False')
        if train:
            if rewards is not None:
                raise AssertionError('Rewards must be None when training')
            self.reset_weights()
            optimizer = Adam(self.parameters(), lr=1e-4)
            writer = SummaryWriter("runs/" + self.model_name, flush_secs=1)
            n_parallel, num_steps = 10, 100
        else:
            num_steps, n_parallel, _ = rewards.shape

        losses = []
        cum_rewards = []
        for i_epoch in range(self.nb_max_epochs if train else 1):
            if train:
                optimizer.zero_grad()
                rewards = generate_task(n_parallel=n_parallel, num_steps=num_steps)
                loss = 0
            cum_reward = np.zeros(n_parallel)
            h = torch.zeros([n_parallel, self.num_units], device=self.device)
            act, rew = torch.randint(2, size=(n_parallel,)), torch.zeros(n_parallel)
            observed_rewards = torch.zeros([num_steps + 1, n_parallel])
            policies = torch.zeros_like(rewards)
            chosen_actions = torch.zeros([num_steps, n_parallel], dtype=int)
            for i_trial in range(rewards.size()[0]):
                pActions, h = self.forward(act, rew, h,n_parallel)
                act = (torch.rand(size=(n_parallel,)) > pActions[:, 0]) * 1
                rew = torch.FloatTensor([rewards[i_trial, i, act[i]] for i in range(n_parallel)])
                cum_reward += rew.detach().numpy() + 0.5
                observed_rewards[i_trial] = rew
                chosen_actions[i_trial] = act
                policies[i_trial] = pActions

            if train:
                discounted_rewards = discount(observed_rewards.numpy(), self.gamma)[:-1]
                loss -= sum([torch.log(policies[i, j, chosen_actions[i, j]]) * discounted_rewards[i, j]
                             for j in range(n_parallel) for i in range(num_steps)])
                entropy = (torch.log(policies + 1e-7) * policies).sum()
                if self.with_entropy:
                    loss += entropy * self.entropy_level
                loss.backward()
                optimizer.step()
                writer.add_scalar('Metrics/Loss', loss, i_epoch)
                writer.add_scalar('Metrics/Entropy', entropy, i_epoch)
                writer.add_scalar('Metrics/Reward', cum_reward.mean(), i_epoch)
                writer.add_scalar('Metrics/control_reward', (rewards + 0.5).mean(axis=0).max(axis=-1).values.mean(), i_epoch)
                losses.append(loss.detach())
            cum_rewards.append(cum_reward.mean())
            if i_epoch % 1000 == 0 and train:
                print('current cumulative rewards : {}'.format(np.mean(cum_rewards[-1000:])))
                torch.save(self.state_dict(), self.path_to_weights / self.model_name)
        cum_rewards = torch.tensor(cum_rewards)

        if plot_results and train:
            from matplotlib import pyplot as plt
            maximum_oracle = torch.max(rewards, axis=1).values.sum()
            plt.figure()
            plt.plot(cum_rewards, label='rnn')
            plt.plot(plt.gca().get_xlim(), [maximum_oracle, maximum_oracle], '--', label='oracle')
            plt.xlabel('epoch number')
            plt.ylabel('rewards')
            plt.legend()
            plt.show()
        return cum_rewards if train else (cum_reward, observed_rewards, chosen_actions)

    def load(self):
        if (self.path_to_weights / self.model_name).exists():
            self.load_state_dict(torch.load(self.path_to_weights / self.model_name))
            print('RNN was loaded')
        else:
            raise AssertionError(f'RNN must be train. '
                                 f'File {(self.path_to_weights / self.model_name).as_posix()} does not exist')

    def scale_W(self, W, multiplier):
        self.load()
        new_value = torch.nn.Parameter(self.__getattr__(W) * multiplier)
        with torch.no_grad():
            self.__setattr__(W, new_value)

if __name__ == '__main__':
    import sys
    try:
        simul_id = int(sys.argv[1]) - 1
    except:
        simul_id = 0
    self = RNN(entropy_level=-1, noise_level=0.5, nb_max_epochs=100000, gamma=0.98, simul_id=simul_id)
    cum_rewards = self.play(rewards=None, train=True, plot_results=True)

    rewards = generate_task()
    a, r = self.play(rewards, train=False)
    print(r.sum())
    print(a)
    print(torch.max(rewards, axis=1).values.sum())
