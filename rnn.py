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
        noise_level=float(-1),
        entropy_level=float(-1),
        nb_max_epochs=1000,
        gamma=float(0.5),
        simul_id=0,
        non_linearity="tanh",
        loaded_existing_model=False,
        training_task="dependent_continuous",
        path_to_weights="weights",
        actor_critic=True,
        two_heads_for_value=False,
        action_dependent_task=False,
    ):
        super(RNN, self).__init__()
        self.num_units = num_units
        self.input_size = input_size
        self.with_noise = noise_level > 0
        self.with_entropy = entropy_level > 0
        self.nb_max_epochs = nb_max_epochs
        self.action_dependent_task = action_dependent_task
        self.gamma = gamma
        self.W_rec = torch.nn.Parameter(torch.randn([self.num_units, self.num_units]))
        self.W_in = torch.nn.Parameter(torch.randn([self.input_size, self.num_units]))
        self.b_rec = torch.nn.Parameter(torch.randn([self.num_units]))
        self.W_out = torch.nn.Parameter(torch.randn([self.num_units, 2]))
        self.two_heads_for_value = two_heads_for_value
        self.noise_level = noise_level
        self.simul_id = simul_id
        self.entropy_level = entropy_level
        self.training_task = training_task
        self.loaded_existing_model = loaded_existing_model
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
        self.model_name = "actorCritic_{}_noise_level_{}_entropy_level_{}_simul_id_{}_inputSize_{}_activation_{}_trainTask_{}_twoHeadsForValue_{}_actionDependentTask_{}.mat".format(
            actor_critic,
            str(self.noise_level).replace(".", "_"),
            str(self.entropy_level).replace(".", "_"),
            str(simul_id),
            str(self.input_size),
            self.non_linearity,
            self.training_task,
            self.two_heads_for_value,
            self.action_dependent_task,
        )

        # Add value head
        self.W_value = torch.nn.Parameter(torch.randn([self.num_units, 1 + self.two_heads_for_value * 1]))

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
        values = torch.zeros([num_steps, n_parallel, 1 + self.two_heads_for_value * 1])
        chosen_actions = torch.zeros([num_steps, n_parallel], dtype=int)

        for i_trial in range(rewards.size()[0]):
            pActions, value, h = self.forward(act, rew, h, n_parallel)
            act = (torch.rand(size=(n_parallel,)).to(self.device) > pActions[:, 0]) * 1
            if self.action_dependent_task:
                nb_times_chosen_arm_has_been_chosen_in_past = (chosen_actions[:i_trial] == act).sum(axis=0)
                rew = rewards[
                    nb_times_chosen_arm_has_been_chosen_in_past, torch.arange(n_parallel), act
                ]
            else:
                rew = rewards[i_trial].gather(1, act.unsqueeze(1)).squeeze(axis=-1)
            cum_reward += (rew.detach().numpy() + 1) / 2.0
            observed_rewards[i_trial] = rew
            chosen_actions[i_trial] = act
            policies[i_trial] = pActions
            values[i_trial] = value

        return cum_reward, observed_rewards, chosen_actions, policies, values.squeeze(-1)

    def play(self, rewards=None, train=True):
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
                discounted_rewards = discount(observed_rewards.numpy(), self.gamma)
                selected_values = values if not self.two_heads_for_value else values.gather(2, chosen_actions.unsqueeze(-1)).squeeze(-1)

                # values_np = selected_values.detach().numpy()
                # advantages = discounted_rewards - values_np * (1. * self.actor_critic)

                advantages = discounted_rewards

                # Policy (actor) loss
                selected_probs = policies.gather(2, chosen_actions.unsqueeze(-1)).squeeze(-1)
                policy_loss = -(torch.log(selected_probs + 1e-7) * torch.tensor(np.ascontiguousarray(advantages))).mean()

                # Value (critic) loss
                value_loss = 0.5 * ((selected_values - torch.tensor(np.ascontiguousarray(discounted_rewards))) ** 2).mean()

                # Entropy loss (if enabled)
                entropy = (torch.log(policies + 1e-7) * policies).mean()

                # Combined loss
                loss = policy_loss + 1 * value_loss * (1. * self.actor_critic) # 10 instead of 0.1
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

        cum_rewards = torch.tensor(cum_rewards)

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

    self = RNN(
        entropy_level=-1,
        noise_level=0.5,
        nb_max_epochs=100000,
        gamma=0.,
        simul_id=2,
        input_size=1,
        non_linearity="tanh",
        training_task="independent_continuous",
        actor_critic=True,
        two_heads_for_value=True,
        action_dependent_task=False,
    )

    cum_rewards = self.play(rewards=None, train=True)

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
    dic["idx_simul"] = self.simul_id
    savemat("/Users/vwyart/Documents/RLCOR/pytorch_conditioning/weights/" + self.model_name, dic)
