import numpy as np
import torch
from scipy.stats import truncnorm
from torch import nn
from torch.optim import Adam, RMSprop
import scipy
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os
from scipy.stats import beta
from generate_task import generate_task


def normalized_columns_initializer(shape, std=1.0):
    out = np.random.randn(*shape).astype(np.float32)
    out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return torch.from_numpy(out)


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class RNN(nn.Module):
    def __init__(
        self,
        num_units=48,
        input_size=3,
        noise_level=-1,
        entropy_level=-1,
        nb_max_epochs=1000,
        gamma=0.5,
        simul_id=0,
        non_linearity="tanh",
        loaded_existing_model=False,
        training_task="dependent_continuous",
        path_to_weights="weights",
        actor_critic=True,
    ):
        super(RNN, self).__init__()
        self.num_units = num_units
        self.input_size = input_size
        self.with_noise = noise_level > 0
        self.with_entropy = entropy_level > 0
        self.nb_max_epochs = nb_max_epochs
        self.gamma = gamma
        self.W_rec = torch.nn.Parameter(torch.randn([self.num_units, self.num_units]))
        self.W_in = torch.nn.Parameter(torch.randn([self.input_size, self.num_units]))
        self.b_rec = torch.nn.Parameter(torch.randn([self.num_units]))
        self.W_out = torch.nn.Parameter(torch.randn([self.num_units, 2]))
        self.noise_level = noise_level
        self.entropy_level = entropy_level
        self.training_task = training_task
        self.loaded_existing_model = loaded_existing_model
        if False and (torch.cuda.is_available() or torch.backends.mps.is_available()):
            print("moving to mps")
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.non_linearity = non_linearity
        if self.non_linearity not in ["relu", "tanh"]:
            raise AssertionError("nonlinearity can only be relu and tanh")
        self.f_non_linearity = (
            torch.nn.Tanh() if self.non_linearity == "tanh" else torch.nn.ReLU()
        )
        self.to(device=self.device)
        self.path_to_weights = Path(path_to_weights)
        self.path_to_weights.mkdir(exist_ok=True)
        self.actor_critic = actor_critic
        self.model_name = "actorCritic_{}_noise_level_{}_entropy_level_{}_simul_id_{}_inputSize_{}_activation_{}_trainTask_{}".format(
            actor_critic,
            str(self.noise_level).replace(".", "_"),
            str(self.entropy_level).replace(".", "_"),
            str(simul_id),
            str(self.input_size),
            self.non_linearity,
            self.training_task,
        )
        
        # Add value head
        self.W_value = torch.nn.Parameter(torch.randn([self.num_units, 1]))
        
    def reset_weights(self):
        with torch.no_grad():
            torch.nn.init.xavier_uniform(self.W_rec)
            torch.nn.init.xavier_uniform(self.W_in)
            self.W_out[:] = normalized_columns_initializer(self.W_out.shape, 0.01)
            self.b_rec[:] = 0
            # Initialize value head weights
            self.W_value[:] = normalized_columns_initializer(self.W_value.shape, 1.0)
            
    def forward(self, prev_act, prev_rew, h, n_parallel):

        if self.input_size == 3:
            input_ = (
                torch.hstack(
                    (
                        prev_rew[:, None].to(self.device),
                        (
                            torch.zeros([len(prev_act), 2]).to(self.device)
                            if torch.all(prev_act == -1)
                            else torch.nn.functional.one_hot(prev_act, num_classes=2)
                        ),
                    )
                )
                .to(torch.float32)
                .to(self.device)
            )
        elif self.input_size == 2:
            input_ = (
                torch.vstack(((1 - prev_act) * prev_rew, prev_act * prev_rew))
                .T.to(torch.float32)
                .to(self.device)
            )
        elif self.input_size == 1:
            input_ = prev_rew[:, None].to(torch.float32).to(self.device)
        h = torch.matmul(input_, self.W_in) + torch.matmul(h, self.W_rec) + self.b_rec
        if self.with_noise:
            h = h + torch.normal(
                mean=0.0,
                std=self.noise_level,
                size=(
                    n_parallel,
                    self.num_units,
                ),
            ).to(self.device)
        h = self.f_non_linearity(h)
        
        # Get policy and value outputs
        policy = torch.nn.Softmax(dim=1)(torch.matmul(h, self.W_out))
        value = torch.matmul(h, self.W_value).squeeze(-1)
        return policy, value, h

    def rnn(self, rewards):
        num_steps, n_parallel, _ = rewards.shape
        cum_reward = np.zeros(n_parallel)
        h = torch.zeros([n_parallel, self.num_units], device=self.device)
        act, rew = torch.zeros(n_parallel, dtype=torch.long) - 1, torch.zeros(n_parallel)
        observed_rewards = torch.zeros([num_steps, n_parallel])
        policies = torch.zeros_like(rewards)
        values = torch.zeros([num_steps, n_parallel])
        chosen_actions = torch.zeros([num_steps, n_parallel], dtype=int)
        
        for i_trial in range(rewards.size()[0]):
            pActions, value, h = self.forward(act, rew, h, n_parallel)
            act = (torch.rand(size=(n_parallel,)).to(self.device) > pActions[:, 0]) * 1
            rew = rewards[i_trial].gather(1, act.unsqueeze(1)).squeeze()
            cum_reward += (rew.detach().numpy() + 1) / 2.0
            observed_rewards[i_trial] = rew
            chosen_actions[i_trial] = act
            policies[i_trial] = pActions
            values[i_trial] = value
            
        return cum_reward, observed_rewards, chosen_actions, policies, values

    def play(self, rewards=None, train=True, plot_results=False):
        if (
            os.path.exists(self.path_to_weights / self.model_name)
            and self.loaded_existing_model
        ):
            self.load_state_dict(
                torch.load(self.path_to_weights / self.model_name), strict=False
            )
            print("model loaded")

        if not train and rewards is None:
            raise AssertionError("reward can not be null if train is False")
        if train:
            if rewards is not None:
                raise AssertionError("Rewards must be None when training")
            self.reset_weights()
            optimizer = RMSprop(self.parameters(), lr=1e-4)
            writer = SummaryWriter("runs/" + self.model_name, flush_secs=1)
            n_parallel, num_steps = 1, 200
        else:
            num_steps, n_parallel, _ = rewards.shape

        losses = []
        cum_rewards = []
        for i_epoch in range(self.nb_max_epochs if train else 1):

            if train:
                optimizer.zero_grad()
                rewards = generate_task(
                    n_parallel=n_parallel,
                    num_steps=num_steps,
                    task=self.training_task,
                )

            cum_reward, observed_rewards, chosen_actions, policies, values = self.rnn(rewards)

            if train:
                # Calculate returns and advantages
                discounted_rewards = discount(observed_rewards.numpy(), self.gamma)[:-1]
                values_np = values.detach().numpy()[:-1]
                advantages = discounted_rewards - values_np * (1. * self.actor_critic)
                
                # Policy (actor) loss
                selected_probs = policies[:-1].gather(2, chosen_actions[:-1].unsqueeze(-1)).squeeze(-1)
                policy_loss = -(torch.log(selected_probs + 1e-7) * torch.tensor(np.ascontiguousarray(advantages))).mean()
                
                # Value (critic) loss
                value_loss = 0.5 * ((values[:-1] - torch.tensor(np.ascontiguousarray(discounted_rewards))) ** 2).mean()
                
                # Entropy loss (if enabled)
                entropy = (torch.log(policies + 1e-7) * policies).mean()
                
                # Combined loss
                loss = policy_loss + 0.5 * value_loss * (1. * self.actor_critic)
                if self.with_entropy:
                    loss += entropy * self.entropy_level
                
                loss.backward()
                optimizer.step()
                
                writer.add_scalar("Metrics/PolicyLoss", policy_loss, i_epoch)
                writer.add_scalar("Metrics/ValueLoss", value_loss, i_epoch)
                writer.add_scalar("Metrics/TotalLoss", loss, i_epoch)
                writer.add_scalar("Metrics/Entropy", entropy, i_epoch)
                writer.add_scalar("Metrics/Reward", cum_reward.mean() / num_steps * 100, i_epoch)
                
                losses.append(loss.detach())
            cum_rewards.append(cum_reward.mean())
            if i_epoch % 1000 == 0 and train:
                torch.save(self.state_dict(), self.path_to_weights / self.model_name)
        
        cum_rewards = torch.tensor(cum_rewards)

        if plot_results and train:
            from matplotlib import pyplot as plt

            maximum_oracle = torch.max(rewards, axis=1).values.sum()
            plt.figure()
            plt.plot(cum_rewards, label="rnn")
            plt.plot(
                plt.gca().get_xlim(),
                [maximum_oracle, maximum_oracle],
                "--",
                label="oracle",
            )
            plt.xlabel("epoch number")
            plt.ylabel("rewards")
            plt.legend()
            plt.show()
        return cum_rewards if train else (cum_reward, observed_rewards, chosen_actions)

    def load(self, path=None, verbose=False):
        if (
            path if path is not None else self.path_to_weights / self.model_name
        ).exists():
            self.load_state_dict(
                torch.load(
                    path if path is not None else self.path_to_weights / self.model_name
                )
            )
            if verbose:
                print("RNN was loaded")
        else:
            raise AssertionError(
                f"RNN must be train. "
                f"File {(self.path_to_weights / self.model_name).as_posix()} does not exist"
            )

    def scale_W(self, W, multiplier):
        self.load()
        new_value = torch.nn.Parameter(self.__getattr__(W) * multiplier)
        with torch.no_grad():
            self.__setattr__(W, new_value)


if __name__ == "__main__":
    import sys

    try:
        run_id = int(sys.argv[1]) - 1
    except:
        run_id = 10

    noise_levels = [0.5]

    nb_noise_levels = len(noise_levels)

    simul_id = int(run_id / nb_noise_levels)
    noise_id = run_id % nb_noise_levels

    self = RNN(
        entropy_level=-1,
        noise_level=noise_levels[noise_id],
        nb_max_epochs=100000,
        gamma=0.5,
        simul_id=simul_id,
        input_size=2,
        non_linearity="tanh",
        training_task="dependent_continuous",
        actor_critic=False,
    )

    cum_rewards = self.play(rewards=None, train=True, plot_results=False)

    # save model
    from scipy.io import savemat
    dic = {}
    dic["weights_rnn_W"] = self.W_rec.detach().numpy()
    dic["weights_rnn_b"] = self.b_rec.detach().numpy()
    dic["weights_input"] = self.W_in.detach().numpy()
    dic["weights_output_policy"] = self.W_out.detach().numpy()
    dic["weights_output_value"] = self.W_value.detach().numpy()
    dic["regul_type"] = "white"
    dic["regul_coeff"] = self.noise_level
    dic["idx_simul"] = simul_id
    savemat("/Users/csmfindling/Documents/Postdoc-Geneva/reliability_VW/theo/pytorch_conditioning/weights/weights_mat/" + self.model_name, dic)


    """
    import dask
    from dask.distributed import Client, progress

    client = Client(threads_per_worker=4, n_workers=4)
    # client.shutdown()

    def train_RNN(args):
        noise_level, simul_id = args
        self = RNN(
            entropy_level=-1,
            noise_level=noise_level,
            nb_max_epochs=100000,
            gamma=0.5,
            simul_id=simul_id,
            input_size=3,
            non_linearity="tanh",
            training_on_restless=False,
        )

        cum_rewards = self.play(rewards=None, train=True, plot_results=False)
        return cum_rewards

    futures = []
    nb_simuls = 15
    noise_levels = [
        0,
        0.2,
        0.5,
        0.8,
        1.2,
        1.5,
    ]  # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.5, 2]
    for i_s in range(nb_simuls):
        for noise_level in noise_levels:
            future = client.submit(train_RNN, [noise_level, i_s])
            futures.append(future)

    results = client.gather(futures)  # futures #
    """
