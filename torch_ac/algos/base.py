from abc import ABC, abstractmethod
import torch

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, eval_envs, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                 mem_type, ext_len, mem_len, n_layer):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        mem_type : string
            the memory type like LSTM or TrXL
        mem_len: int
            the length of memory
        n_layer : int
            layers of memory module
        """

        # Store parameters

        self.num_eval_procs = len(eval_envs)
        self.eval_env = ParallelEnv(eval_envs)
        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.mem_type = mem_type
        self.ext_len = ext_len
        self.mem_len = mem_len
        self.n_layer = n_layer

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        img_shape = self.obs[0]['image'].shape
        self.obss = [None]*(shape[0])
        if self.acmodel.recurrent:
            if self.mem_type == 'lstm':
                self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
                self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
            else:  # transformers
                self.memory = torch.zeros(shape[1], self.n_layer+1, self.mem_len, self.acmodel.semi_memory_size, device=self.device)
                self.memories = torch.zeros(*shape, self.n_layer+1, self.mem_len, self.acmodel.semi_memory_size, device=self.device)
                self.ext_img = torch.zeros(shape[1], self.ext_len, *img_shape, device=self.device)
                self.ext_imgs = torch.zeros(*shape, self.ext_len, *img_shape, device=self.device)
                self.ext_act = torch.zeros(shape[1], self.ext_len, device=self.device)
                self.ext_acts = torch.zeros(*shape, self.ext_len, device=self.device)
                self.ext_reward = torch.zeros(shape[1], self.ext_len, device=self.device)
                self.ext_rewards = torch.zeros(*shape, self.ext_len, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        self.prev_actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.prev_rewards = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            self.prev_actions[i] = prev_action = self.actions[i-1]
            self.prev_rewards[i] = prev_reward = self.rewards[i-1]
            with torch.no_grad():
                if self.acmodel.recurrent:
                    if self.mem_type == 'lstm':
                        dist, value, memory, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),
                                prev_action * self.mask, prev_reward * self.mask)
                    elif 'trxl' in self.mem_type:  # transformers
                        dist, value, memory, ext = self.acmodel(preprocessed_obs,
                                (self.memory*self.mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).permute(1,2,0,3),
                                prev_action * self.mask, prev_reward * self.mask,
                                ext_img=self.ext_img*self.mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                                ext_act=self.ext_act*self.mask.unsqueeze(-1), ext_reward=self.ext_reward*self.mask.unsqueeze(-1))
                    else:
                        raise ValueError(f"The type must be one of lstm or trxls.")
                else:
                    dist, value, _, _ = self.acmodel(preprocessed_obs)
            action = dist.sample()

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
                if 'trxl' in self.mem_type:
                    self.ext_imgs[i] = self.ext_img
                    self.ext_img = ext['image']
                    self.ext_acts[i] = self.ext_act
                    self.ext_act = ext['action']
                    self.ext_rewards[i] = self.ext_reward
                    self.ext_reward = ext['reward']
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                if self.mem_type =='lstm':
                    _, next_value, _, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),
                            self.actions[-1] * self.mask, self.rewards[-1] * self.mask)
                else:  # transformers
                    _, next_value, _, _ = self.acmodel(preprocessed_obs,
                            (self.memory*self.mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).permute(1,2,0,3),
                            self.actions[-1] * self.mask, self.rewards[-1] * self.mask,
                            ext_img=self.ext_img*self.mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                            ext_act=self.ext_act*self.mask.unsqueeze(-1), ext_reward=self.ext_reward*self.mask.unsqueeze(-1))
            else:
                _, next_value, _, _ = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T)
            exps.mask = self.masks.transpose(0, 1).reshape(-1)
            if 'trxl' in self.mem_type:
                # T x P x D -> P x T x D -> (P * T) x D
                exps.ext_img = self.ext_imgs.transpose(0, 1).reshape(-1, *self.ext_imgs.shape[2:])
                exps.ext_act = self.ext_acts.transpose(0, 1).reshape(-1, *self.ext_acts.shape[2:])
                exps.ext_reward = self.ext_rewards.transpose(0, 1).reshape(-1, *self.ext_rewards.shape[2:])
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        exps.prev_action = self.prev_actions.transpose(0, 1).reshape(-1)
        exps.prev_reward = self.prev_rewards.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    # test agent
    def run_evaluation(self):

        total_reward = 0.0
        obs = self.eval_env.reset()
        img_shape = obs[0]['image'].shape
        action = torch.zeros(self.num_eval_procs, device=self.device)
        reward = torch.zeros(self.num_eval_procs, device=self.device)
        total_reward = torch.zeros(self.num_eval_procs, device=self.device)
        mask = torch.ones(self.num_eval_procs, device=self.device)
        all_done = False
        if self.mem_type == 'lstm':
            memory = torch.zeros(self.num_eval_procs, self.acmodel.memory_size, device=self.device)
        else:
            memory = torch.zeros(self.num_eval_procs, self.n_layer+1, self.mem_len, self.acmodel.semi_memory_size, device=self.device)
            ext_img = torch.zeros(self.num_eval_procs, self.ext_len, *img_shape, device=self.device)
            ext_act = torch.zeros(self.num_eval_procs, self.ext_len, device=self.device)
            ext_reward = torch.zeros(self.num_eval_procs, self.ext_len, device=self.device)

        while not all_done:
            preprocessed_obs = self.preprocess_obss(obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    if self.mem_type == 'lstm':
                        dist, value, memory, _ = self.acmodel(preprocessed_obs, memory, action, reward)
                    elif 'trxl' in self.mem_type:  # transformers
                        dist, value, memory, ext = self.acmodel(preprocessed_obs,
                                memory.permute(1,2,0,3),
                                action, reward,
                                ext_img=ext_img, ext_act=ext_act, ext_reward=ext_reward)
                    else:
                        raise ValueError(f"The type must be one of lstm or trxls.")
                else:
                    dist, value, _, _ = self.acmodel(preprocessed_obs)
            action = dist.probs.max(1)[1]

            obs, reward, done, _ = self.eval_env.step(action.cpu().numpy())

            if self.acmodel.recurrent:
                if 'trxl' in self.mem_type:
                    ext_img = ext['image']
                    ext_act = ext['action']
                    ext_reward = ext['reward']
            if self.reshape_reward is not None:
                reward = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                reward = torch.tensor(reward, dtype=torch.float, device=self.device)

            total_reward += reward * mask

            for idx, _d in enumerate(done):
                if _d:
                    mask[idx] = 0

            if mask.sum().item() == 0:
                all_done=True

        return total_reward.mean()

    @abstractmethod
    def update_parameters(self):
        pass
