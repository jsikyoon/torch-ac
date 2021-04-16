import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class DREAMERAlgo(BaseAlgo):
    """The Dreamer algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, mem_type='lstm', mem_len=10, n_layer=5, loss_type='rep-img',
                 combine_loss=False, lr_rep=0.001, lr_img=8e-5, n_imagine=5, use_real=True,
                 img_epochs=5, rep_epochs=50):

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         mem_type, mem_len, n_layer)

        self.clip_eps = clip_eps
        self.batch_size = batch_size
        self.mem_type = mem_type
        self.loss_type = loss_type.split('-')
        self.combine_loss = combine_loss
        self.n_imagine = n_imagine
        self.use_real = use_real
        self.img_epochs = img_epochs
        self.rep_epochs = rep_epochs
        self.epochs = max(img_epochs, rep_epochs)
        self.batch_num = 0

        assert self.batch_size % self.recurrence == 0

        self.rep_optimizer = torch.optim.Adam(self.acmodel.parameters(), lr_rep, eps=adam_eps)
        self.img_optimizer = torch.optim.Adam(self.acmodel.parameters(), lr_img, eps=adam_eps)

    def update_parameters(self, exps):
        # Collect experiences

        for epoch in range(self.epochs):

            if epoch < self.rep_epochs:

                ################################################################
                # Representation of Dreamer
                ################################################################

                log_rep_losses = []
                log_recon_accs = []
                log_recon_losses = []
                log_recon_col_losses = []
                log_recon_obj_losses = []
                log_recon_state_losses = []
                log_reward_mse = []
                log_reward_logprob = []
                log_nonzero_reward_mse = []
                log_nonzero_reward_logprob = []
                log_nonzero_reward_num = []
                log_zero_reward_mse = []
                log_zero_reward_logprob = []
                log_zero_reward_num = []
                log_kl_losses = []

                for inds in self._get_batches_starting_indexes(self.recurrence):

                    # Initialize batch values

                    update_rep_loss = 0
                    update_recon_acc = 0
                    update_recon_loss = 0
                    update_recon_col_loss = 0
                    update_recon_obj_loss = 0
                    update_recon_state_loss = 0
                    update_reward_mse = 0
                    update_reward_logprob = 0
                    update_nonzero_reward_mse = 0
                    update_nonzero_reward_logprob = 0
                    update_nonzero_reward_num = 0
                    update_zero_reward_mse = 0
                    update_zero_reward_logprob = 0
                    update_zero_reward_num = 0
                    update_kl_loss = 0

                    # Initialize memory

                    if self.acmodel.recurrent:
                        memory = exps.memory[inds]
                        prev_state = exps.prev_state[inds]

                    for i in range(self.recurrence):
                        # Create a sub-batch of experience

                        sb = exps[inds + i]
                        prev_action = exps.prev_action[inds + i]
                        action = exps.action[inds + i]

                        # Compute loss

                        if self.acmodel.recurrent:
                            if self.mem_type == 'lstm':
                                dist, value, memory, state, rep_loss = self.acmodel(sb.obs, memory * sb.mask,
                                        prev_action * sb.mask[:,0],
                                        prev_state * sb.mask,
                                        action,
                                        sb.reward,
                                        get_rep_loss=True)
                            else: # transformers
                                dist, value, memory, state, rep_loss  = self.acmodel(sb.obs, (memory*sb.mask).permute(1,2,0,3),
                                        prev_action * sb.mask[:,0,0,0],
                                        prev_state * sb.mask[:,:,0,0],
                                        action,
                                        sb.reward,
                                        get_rep_loss=True)
                        else:
                            #dist, value = self.acmodel(sb.obs)
                            raise ValueError("Dreamer is memory-based model.")

                        prev_state = state

                        # Update representation losses

                        update_rep_loss += rep_loss['rep_loss']
                        update_recon_acc += rep_loss['recon_acc']
                        update_recon_loss += rep_loss['recon_loss'].item()
                        update_recon_col_loss += rep_loss['recon_col_loss'].item()
                        update_recon_obj_loss += rep_loss['recon_obj_loss'].item()
                        update_recon_state_loss += rep_loss['recon_state_loss'].item()
                        update_reward_mse += rep_loss['reward_mse']
                        update_reward_logprob += rep_loss['reward_logprob']
                        update_nonzero_reward_mse += rep_loss['nonzero_reward_mse']
                        update_nonzero_reward_logprob += rep_loss['nonzero_reward_logprob']
                        update_nonzero_reward_num += rep_loss['nonzero_reward_num']
                        update_zero_reward_mse += rep_loss['zero_reward_mse']
                        update_zero_reward_logprob += rep_loss['zero_reward_logprob']
                        update_zero_reward_num += rep_loss['zero_reward_num']
                        update_kl_loss += rep_loss['kl_loss'].item()

                        # Update memories for next epoch

                        if 'trxl' in self.mem_type:
                            memory = torch.stack(memory,dim=0).permute(2,0,1,3)
                        if i < self.recurrence - 1:
                            exps.memory[inds + i + 1] = memory.detach()
                            exps.prev_state[inds + i + 1] = state.detach()
                        else:
                            _inds = []
                            _memory = []
                            _state = []
                            for j in range(len(inds)):
                                if inds[j] + i + 1  < len(exps.memory):
                                    _inds.append(inds[j])
                                    _memory.append(memory[j].detach())
                                    _state.append(state[j].detach())
                            _inds = numpy.array(_inds)
                            exps.memory[_inds + i + 1] = torch.stack(_memory, dim=0)
                            exps.prev_state[_inds + i + 1] = torch.stack(_state, dim=0)

                    # Update update values

                    update_rep_loss /= self.recurrence
                    update_recon_acc /= self.recurrence
                    update_recon_loss /= self.recurrence
                    update_recon_col_loss /= self.recurrence
                    update_recon_obj_loss /= self.recurrence
                    update_recon_state_loss /= self.recurrence
                    update_reward_mse /= self.recurrence
                    update_reward_logprob /= self.recurrence
                    if update_nonzero_reward_num != 0:
                        update_nonzero_reward_mse /= update_nonzero_reward_num
                        update_nonzero_reward_logprob /= update_nonzero_reward_num
                    else:
                        update_nonzero_reward_mse = float('nan')
                        update_nonzero_reward_logprob = float('nan')
                    #update_nonzero_reward_num /= self.recurrence
                    if update_zero_reward_num != 0:
                        update_zero_reward_mse /= update_zero_reward_num
                        update_zero_reward_logprob /= update_zero_reward_num
                    else:
                        update_zero_reward_mse = float('nan')
                        update_zero_reward_logprob = float('nan')
                    #update_zero_reward_num /= self.recurrence
                    update_kl_loss /= self.recurrence

                # Update representation modules

                self.rep_optimizer.zero_grad()
                update_rep_loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.clip_dreamer)
                self.rep_optimizer.step()

                # Update log values

                log_rep_losses.append(update_rep_loss.item())
                log_recon_accs.append(update_recon_acc)
                log_recon_losses.append(update_recon_loss)
                log_recon_col_losses.append(update_recon_col_loss)
                log_recon_obj_losses.append(update_recon_obj_loss)
                log_recon_state_losses.append(update_recon_state_loss)
                log_reward_mse.append(update_reward_mse.item())
                log_reward_logprob.append(update_reward_logprob.item())
                log_nonzero_reward_mse.append(update_nonzero_reward_mse)
                log_nonzero_reward_logprob.append(update_nonzero_reward_logprob)
                log_nonzero_reward_num.append(update_nonzero_reward_num)
                log_zero_reward_mse.append(update_zero_reward_mse)
                log_zero_reward_logprob.append(update_zero_reward_logprob)
                log_zero_reward_num.append(update_zero_reward_num)
                log_kl_losses.append(update_kl_loss)


            if epoch < self.img_epochs:

                ################################################################
                # Agent in imagination
                ################################################################

                imagination = self.n_imagine

                # Compute starting indexes

                #inds = numpy.arange(0, self.num_frames, imagination)
                inds = numpy.arange(0, self.num_frames, 1) # to get the dist of prev

                # Initialize log values

                log_entropies = []
                log_values = []
                log_policy_losses = []
                log_value_losses = []
                log_grad_norms = []

                # Initialize update values

                update_entropy = 0
                update_value = 0
                update_policy_loss = 0
                update_value_loss = 0
                update_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]
                    prev_state = exps.prev_state[inds]

                sbs = []
                dists = []
                values = []
                states = []
                actions = []
                rewards = []

                # Create a sub-batch of experience

                sb = exps[inds]
                prev_action = exps.prev_action[inds]

                # Compute loss
                if self.acmodel.recurrent:
                    if self.mem_type == 'lstm':
                        dist, value, memory, state, _ = self.acmodel(sb.obs, memory * sb.mask,
                                prev_action * sb.mask[:,0],
                                prev_state * sb.mask)
                    else: # transformers
                        dist, value, memory, state, _ = self.acmodel(sb.obs,
                                (memory*sb.mask).permute(1,2,0,3),
                                prev_action * sb.mask[:,0,0,0],
                                prev_state * sb.mask[:,:,0,0])
                else:
                    #dist, value = self.acmodel(sb.obs)
                    raise ValueError("Dreamer is memory-based model.")

                sbs.append(sb)
                dists.append(dist)
                values.append(value)
                states.append(state)
                if self.use_real:
                    actions.append(exps.action[inds])
                    rewards.append(exps.reward[inds])
                else:
                    actions.append(dist.sample())
                    rewards.append(self.acmodel.reward_decoder(
                        torch.cat([states[-1], actions[-1].unsqueeze(-1)], dim=-1)).squeeze())

                # Update memories for next epoch

                if 'trxl' in self.mem_type:
                    memory = torch.stack(memory,dim=0).permute(2,0,1,3)
                _inds = []
                _memory = []
                _state = []
                for j in range(len(inds)):
                    if inds[j]+i+1  < len(exps.memory):
                        _inds.append(inds[j])
                        _memory.append(memory[j].detach())
                        _state.append(state[j].detach())
                _inds = numpy.array(_inds)
                exps.memory[_inds + i + 1] = torch.stack(_memory, dim=0)
                exps.prev_state[_inds + i + 1] = torch.stack(_state, dim=0)

                for i in range(1,imagination):
                    # Create a sub-batch of experience

                    #sb = exps[inds + i]
                    sb = exps[inds]  # dummy input (not used in function)
                    #action = exps.prev_action[inds + i]

                    # Compute loss
                    if self.acmodel.recurrent:
                        if self.mem_type == 'lstm':
                            dist, value, memory, state, _ = self.acmodel(sb.obs, memory,
                                    actions[-1],
                                    states[-1],
                                    get_prior=True)
                        else: # transformers
                            dist, value, memory, state, _ = self.acmodel(sb.obs,
                                    memory.permute(1,2,0,3),
                                    actions[-1],
                                    states[-1],
                                    get_prior=True)
                            memory = torch.stack(memory,dim=0).permute(2,0,1,3)
                    else:
                        #dist, value = self.acmodel(sb.obs)
                        raise ValueError("Dreamer is memory-based model.")

                    sbs.append(sb)
                    dists.append(dist)
                    values.append(value)
                    states.append(state)
                    actions.append(dist.sample())
                    rewards.append(self.acmodel.reward_decoder(
                        torch.cat([states[-1], actions[-1].unsqueeze(-1)], dim=-1)).squeeze())

                # get V_lambda
                img_Vs = []
                gamma = 0.99
                _lambda = 0.95
                for i in range(imagination):
                    _Vns = [0] # first element is not used
                    for j in range(1, imagination):
                        h = min(i+j, imagination-1)
                        _Vn = 0
                        for k in range(i,h):
                            _Vn += gamma**(k-i)*rewards[k]
                        _Vn += gamma**(h-i)*values[h]
                        _Vns.append(_Vn)
                    _img_V = 0
                    for j in range(1, imagination-1):
                        _img_V += _lambda**(j-1)*_Vns[j]
                    img_Vs.append((1-_lambda)*_img_V + _lambda**(imagination-1)*_Vns[-1])

                for i in range(imagination):

                    entropy = dists[i].entropy().mean()

                    #policy_loss = -(dists[i].log_prob(sbs[i].action) * sb.advantage).mean()
                    #policy_loss = -(dists[i].log_prob(sbs[i].action) * img_Vs[i].detach()).mean()
                    policy_loss = -(dists[i].log_prob(actions[i]) * img_Vs[i].detach()).mean()

                    value_loss = (values[i] - img_Vs[i]).pow(2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    update_entropy += entropy.item()
                    update_value += value.mean().item()
                    update_policy_loss += policy_loss.item()
                    update_value_loss += value_loss.item()
                    update_loss += loss

                # Update update values

                update_entropy /= imagination
                update_value /= imagination
                update_policy_loss /= imagination
                update_value_loss /= imagination
                update_loss /= imagination

                # Update actor-critic

                self.img_optimizer.zero_grad()
                update_loss.backward()
                grad_norm = 0
                for p in self.acmodel.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                #grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.img_optimizer.step()

                # Update log values

                log_entropies.append(update_entropy)
                log_values.append(update_value)
                log_policy_losses.append(update_policy_loss)
                log_value_losses.append(update_value_loss)
                log_grad_norms.append(grad_norm)


        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),

            "rep_loss": numpy.mean(log_rep_losses),

            "recon_acc": numpy.mean(log_recon_accs),
            "recon_loss": numpy.mean(log_recon_losses),
            "recon_col_loss": numpy.mean(log_recon_col_losses),
            "recon_obj_loss": numpy.mean(log_recon_obj_losses),
            "recon_state_loss": numpy.mean(log_recon_state_losses),

            "reward_mse": numpy.mean(log_reward_mse),
            "reward_logprob": numpy.mean(log_reward_logprob),
            "nonzero_reward_mse": numpy.mean(log_nonzero_reward_mse),
            "nonzero_reward_logprob": numpy.mean(log_nonzero_reward_logprob),
            "nonzero_reward_num": numpy.sum(log_nonzero_reward_num),
            "zero_reward_mse": numpy.mean(log_zero_reward_mse),
            "zero_reward_logprob": numpy.mean(log_zero_reward_logprob),
            "zero_reward_num": numpy.sum(log_zero_reward_num),

            "kl_loss": numpy.mean(log_kl_losses),
        }

        return logs

    def _get_batches_starting_indexes(self, recurrence):
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
        indexes = numpy.arange(0, self.num_frames, recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + recurrence) % self.num_frames_per_proc != 0]
            indexes += recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
