import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None,
                 mem_type='lstm', mem_len=10, n_layer=5, loss_type='all'):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         mem_type, mem_len, n_layer)

        self.mem_type = mem_type
        self.loss_type = loss_type

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        update_rep_loss = 0
        update_recon_loss = 0
        update_reward_loss = 0
        update_kl_loss = 0

        update_img_loss = 0
        update_img_policy_loss = 0
        update_img_value_loss = 0

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]
            action = exps.prev_action[inds]
            state = exps.prev_state[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Compute loss
            if self.acmodel.recurrent:
                if self.mem_type == 'lstm':
                    dist, value, memory, state, rep_loss, img_loss = self.acmodel(sb.obs, memory * sb.mask,
                            action * sb.mask[:,0],
                            state * sb.mask,
                            sb.reward,
                            get_rep_loss=True,
                            get_img_loss=True)
                else: # transformers
                    dist, value, memory, state, rep_loss, img_loss = self.acmodel(sb.obs,
                            (memory*sb.mask).permute(1,2,0,3),
                            action * sb.mask[:,0,0,0],
                            state * sb.mask[:,:,0,0],
                            sb.reward,
                            get_rep_loss=True,
                            get_img_loss=True)
            else:
                #dist, value = self.acmodel(sb.obs)
                raise ValueError("Dreamer is memory-based model.")

            action = dist.sample()

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            if self.loss_type in ['all', 'rep-agent']:
                update_loss += loss

            # Update representation losses

            update_rep_loss += rep_loss['rep_loss'].item()
            update_recon_loss += rep_loss['recon_loss'].item()
            update_reward_loss += rep_loss['reward_loss'].item()
            update_kl_loss += rep_loss['kl_loss'].item()
            update_loss += rep_loss['rep_loss']

            # Update imaginary losses

            update_img_loss += img_loss['img_loss'].item()
            update_img_policy_loss += img_loss['policy_loss'].item()
            update_img_value_loss += img_loss['value_loss'].item()
            if self.loss_type in ['all', 'rep-img']:
                update_loss += img_loss['img_loss']

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        update_rep_loss /= self.recurrence
        update_recon_loss /= self.recurrence
        update_reward_loss /= self.recurrence
        update_kl_loss /= self.recurrence

        update_img_loss /= self.recurrence
        update_img_policy_loss /= self.recurrence
        update_img_value_loss /= self.recurrence

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm,

            "rep_loss": update_rep_loss,
            "recon_loss": update_recon_loss,
            "reward_loss": update_reward_loss,
            "kl_loss": update_kl_loss,

            "img_loss": update_img_loss,
            "img_policy_loss": update_img_policy_loss,
            "img_value_loss": update_img_value_loss,
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
