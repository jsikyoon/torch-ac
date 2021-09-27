import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class VMPOAlgo(BaseAlgo):

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, mem_type='lstm', ext_len=10, mem_len=10, n_layer=5,
                 img_encode=False):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         mem_type, ext_len, mem_len, n_layer, img_encode)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.mem_type = mem_type

        self.eta = torch.autograd.Variable(torch.tensor(1.0), requires_grad=True)
        self.alpha = torch.autograd.Variable(torch.tensor(0.1), requires_grad=True)
        self.eps_eta = 0.02
        self.eps_alpha = 0.1

        assert self.batch_size % self.recurrence == 0

        self.params = list(self.acmodel.parameters()) + [self.eta, self.alpha]
        self.optimizer = torch.optim.Adam(self.params, lr, eps=adam_eps)
        self.batch_num = 0

        self.MseLoss = nn.MSELoss()

    def get_KL(self, prob1, logprob1, logprob2):
        kl = prob1 * (logprob1 - logprob2)
        return kl

    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        if self.mem_type == 'lstm':
                            dist, value, memory, _ = self.acmodel(sb.obs, memory * sb.mask.unsqueeze(1),
                                sb.prev_action*sb.mask, sb.prev_reward*sb.mask)
                        else: # transformers
                            dist, value, memory, ext = self.acmodel(sb.obs,
                                (memory*sb.mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).permute(1,2,0,3),
                                sb.prev_action*sb.mask, sb.prev_reward*sb.mask,
                                ext_img=sb.ext_img*sb.mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                                ext_act=sb.ext_act*sb.mask.unsqueeze(-1), ext_reward=sb.ext_reward*sb.mask.unsqueeze(-1))
                    else:
                        dist, value, _, _ = self.acmodel(sb.obs)

                    # Get samples with top half advantages
                    advprobs = torch.stack((sb.advantage, dist.log_prob(sb.action)))
                    advprobs = advprobs[:, torch.sort(advprobs[0], descending=True).indices]
                    good_advantages = advprobs[0, :advprobs.shape[1]//2]
                    good_logprobs = advprobs[1, :advprobs.shape[1]//2]

                    # Get losses
                    phis = torch.exp(good_advantages/self.eta.detach())/torch.sum(torch.exp(good_advantages/self.eta.detach()))
                    L_pi = -phis*good_logprobs
                    L_eta = self.eta*self.eps_eta+self.eta*torch.log(torch.mean(torch.exp(good_advantages/self.eta)))

                    KL = self.get_KL(torch.exp(sb.log_prob), sb.log_prob, dist.log_prob(sb.action))

                    L_alpha = torch.mean(self.alpha*(self.eps_alpha-KL.detach())+self.alpha.detach()*KL)

                    loss = L_pi + L_eta + L_alpha + 0.5*self.MseLoss(value, sb.returnn)

                    # Update batch values

                    #batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    #batch_policy_loss += policy_loss.item()
                    #batch_value_loss += value_loss.item()
                    batch_loss += loss.mean()

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()
                        if 'trxl' in self.mem_type:
                            exps.ext[inds + i + 1] = ext.detach()

                # Update batch values

                #batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                #batch_policy_loss /= self.recurrence
                #batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.params) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    self.eta.copy_(torch.clamp(self.eta, min=1e-8))
                    self.alpha.copy_(torch.clamp(self.alpha, min=1e-8))

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
