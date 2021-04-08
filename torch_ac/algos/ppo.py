import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, mem_type='lstm', mem_len=10, n_layer=5, loss_type='agent-rep-img',
                 combine_loss=False,
                 lr_rep=0.001, lr_img=8e-5, clip_dreamer=100.0, n_imagine=5, img_method='ppo'):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         mem_type, mem_len, n_layer)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.mem_type = mem_type
        self.loss_type = loss_type.split('-')
        self.combine_loss = combine_loss
        self.clip_dreamer = clip_dreamer
        self.n_imagine = n_imagine
        self.img_method = img_method

        assert self.batch_size % self.recurrence == 0

        self.agent_optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.rep_optimizer = torch.optim.Adam(self.acmodel.parameters(), lr_rep)
        self.img_optimizer = torch.optim.Adam(self.acmodel.parameters(), lr_img)
        self.batch_num = 0

    def update_parameters(self, exps):
        # Collect experiences

        ################################################################
        # Agent (ppo)
        ################################################################

        # set recurrence = 1 when doesn't train representation modules on here

        if ('rep' in self.loss_type) and (not self.combine_loss):
            recurrence = 1
        else:
            recurrence = self.recurrence

        for _ in range(self.epochs):

            # do not update through ppo algorithm

            if not 'agent' in self.loss_type:
                log_entropies = [0]
                log_values = [0]
                log_policy_losses = [0]
                log_value_losses = [0]
                log_grad_norms = [0]
                break

            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes(recurrence):
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]
                    state = exps.prev_state[inds]

                for i in range(recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]
                    action = exps.prev_action[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        if self.mem_type == 'lstm':
                            dist, value, memory, state, _ = self.acmodel(sb.obs, memory * sb.mask,
                                    action * sb.mask[:,0],
                                    state * sb.mask)
                        else: # transformers
                            dist, value, memory, state, _ = self.acmodel(sb.obs,
                                    (memory*sb.mask).permute(1,2,0,3),
                                    action * sb.mask[:,0,0,0],
                                    state * sb.mask[:,:,0,0])
                    else:
                        #dist, value = self.acmodel(sb.obs)
                        raise ValueError("Dreamer is memory-based model.")

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < recurrence - 1:
                        if 'trxl' in self.mem_type:
                            memory = torch.stack(memory,dim=0).permute(2,0,1,3)
                        exps.memory[inds + i + 1] = memory.detach()
                        exps.prev_state[inds + i + 1] = state.detach()

                # Update batch values

                batch_entropy /= recurrence
                batch_value /= recurrence
                batch_policy_loss /= recurrence
                batch_value_loss /= recurrence
                batch_loss /= recurrence

                # Update actor-critic

                self.agent_optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = 0
                for p in self.acmodel.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                #grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.agent_optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        ################################################################
        # Agent in imagination
        ################################################################

        imagination = self.n_imagine

        if not 'img' in self.loss_type:
            img_method = None
        else:
            img_method = self.img_method

        if img_method == 'ppo':

            for _ in range(self.epochs):

                # Initialize log values

                log_entropies = []
                log_values = []
                log_policy_losses = []
                log_value_losses = []
                log_grad_norms = []

                for inds in self._get_batches_starting_indexes(imagination):
                    # Initialize batch values

                    batch_entropy = 0
                    batch_value = 0
                    batch_policy_loss = 0
                    batch_value_loss = 0
                    batch_loss = 0

                    # Initialize memory

                    if self.acmodel.recurrent:
                        memory = exps.memory[inds]
                        state = exps.prev_state[inds]

                    for i in range(imagination):
                        # Create a sub-batch of experience

                        sb = exps[inds + i]
                        action = exps.prev_action[inds + i]

                        # Compute loss

                        if self.acmodel.recurrent:
                            if self.mem_type == 'lstm':
                                dist, value, memory, state, _ = self.acmodel(sb.obs,
                                        memory * sb.mask,
                                        action * sb.mask[:,0],
                                        state * sb.mask,
                                        get_prior=True)
                            else: # transformers
                                dist, value, memory, state, _ = self.acmodel(sb.obs,
                                        (memory*sb.mask).permute(1,2,0,3),
                                        action * sb.mask[:,0,0,0],
                                        state * sb.mask[:,:,0,0],
                                        get_prior=True)
                        else:
                            #dist, value = self.acmodel(sb.obs)
                            raise ValueError("Dreamer is memory-based model.")

                        entropy = dist.entropy().mean()

                        ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                        surr1 = ratio * sb.advantage
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                        surr1 = (value - sb.returnn).pow(2)
                        surr2 = (value_clipped - sb.returnn).pow(2)
                        value_loss = torch.max(surr1, surr2).mean()

                        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                        # Update batch values

                        batch_entropy += entropy.item()
                        batch_value += value.mean().item()
                        batch_policy_loss += policy_loss.item()
                        batch_value_loss += value_loss.item()
                        batch_loss += loss

                        # Update memories for next epoch

                        if self.acmodel.recurrent and i < imagination - 1:
                            if 'trxl' in self.mem_type:
                                memory = torch.stack(memory,dim=0).permute(2,0,1,3)
                            exps.memory[inds + i + 1] = memory.detach()
                            exps.prev_state[inds + i + 1] = state.detach()

                    # Update batch values

                    batch_entropy /= imagination
                    batch_value /= imagination
                    batch_policy_loss /= imagination
                    batch_value_loss /= imagination
                    batch_loss /= imagination

                    # Update actor-critic

                    self.img_optimizer.zero_grad()
                    batch_loss.backward()
                    grad_norm = 0
                    for p in self.acmodel.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    #grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                    self.img_optimizer.step()

                    # Update log values

                    log_entropies.append(batch_entropy)
                    log_values.append(batch_value)
                    log_policy_losses.append(batch_policy_loss)
                    log_value_losses.append(batch_value_loss)
                    log_grad_norms.append(grad_norm)

        elif img_method == 'a2c':

            # Compute starting indexes

            inds = numpy.arange(0, self.num_frames, imagination)

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
                state = exps.prev_state[inds]

            for i in range(imagination):
                # Create a sub-batch of experience

                sb = exps[inds + i]
                action = exps.prev_action[inds + i]

                # Compute loss
                if self.acmodel.recurrent:
                    if self.mem_type == 'lstm':
                        dist, value, memory, state, _ = self.acmodel(sb.obs, memory * sb.mask,
                                action * sb.mask[:,0],
                                state * sb.mask,
                                get_prior=True)
                    else: # transformers
                        dist, value, memory, state, _ = self.acmodel(sb.obs,
                                (memory*sb.mask).permute(1,2,0,3),
                                action * sb.mask[:,0,0,0],
                                state * sb.mask[:,:,0,0],
                                get_prior=True)
                else:
                    #dist, value = self.acmodel(sb.obs)
                    raise ValueError("Dreamer is memory-based model.")

                entropy = dist.entropy().mean()

                policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

                value_loss = (value - sb.returnn).pow(2).mean()

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

        elif img_method == 'dreamer':

            # Compute starting indexes

            inds = numpy.arange(0, self.num_frames, imagination)

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
                state = exps.prev_state[inds]

            sbs = []
            dists = []
            values = []
            states = []
            actions = []
            rewards = []

            # Create a sub-batch of experience

            sb = exps[inds]
            action = exps.prev_action[inds]

            # Compute loss
            if self.acmodel.recurrent:
                if self.mem_type == 'lstm':
                    dist, value, memory, state, _ = self.acmodel(sb.obs, memory * sb.mask,
                            action * sb.mask[:,0],
                            state * sb.mask)
                else: # transformers
                    dist, value, memory, state, _ = self.acmodel(sb.obs,
                            (memory*sb.mask).permute(1,2,0,3),
                            action * sb.mask[:,0,0,0],
                            state * sb.mask[:,:,0,0])
            else:
                #dist, value = self.acmodel(sb.obs)
                raise ValueError("Dreamer is memory-based model.")

            sbs.append(sb)
            dists.append(dist)
            values.append(value)
            states.append(state)
            actions.append(action)
            rewards.append(exps.reward[inds])

            for i in range(1, imagination):
                # Create a sub-batch of experience

                sb = exps[inds + i]
                action = exps.prev_action[inds + i]

                # Compute loss
                if self.acmodel.recurrent:
                    if self.mem_type == 'lstm':
                        dist, value, memory, state, _ = self.acmodel(sb.obs, memory * sb.mask,
                                action * sb.mask[:,0],
                                state * sb.mask,
                                get_prior=True)
                    else: # transformers
                        dist, value, memory, state, _ = self.acmodel(sb.obs,
                                (memory*sb.mask).permute(1,2,0,3),
                                action * sb.mask[:,0,0,0],
                                state * sb.mask[:,:,0,0],
                                get_prior=True)
                else:
                    #dist, value = self.acmodel(sb.obs)
                    raise ValueError("Dreamer is memory-based model.")

                sbs.append(sb)
                dists.append(dist)
                values.append(value)
                states.append(state)
                actions.append(action)
                rewards.append(exps.reward[inds + i])

            """
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
            """

            # make advantages
            advantages = [0]*(imagination-1)
            returnn = [0]*(imagination-1)
            discount = 0.99
            gae_lambda = 0.95
            for i in reversed(range(imagination-1)):
                if self.mem_type == 'lstm':
                    next_mask = sbs[i+1].mask[:,0]
                else:
                    next_mask = sbs[i+1].mask[:,0,0,0]
                next_value = values[i+1]
                next_advantage = advantages[i+1] if i < imagination - 2 else 0

                delta = rewards[i] + discount * next_value * next_mask - values[i]
                advantages[i] = delta + discount * gae_lambda * next_advantage * next_mask
                returnn[i] = values[i] + advantages[i]

            for i in range(imagination-1):

                entropy = dists[i].entropy().mean()

                policy_loss = -(dists[i].log_prob(sbs[i].action) * advantages[i].detach()).mean()
                #policy_loss = -(dists[i].log_prob(sbs[i].action) * img_Vs[i].detach()).mean()

                value_loss = (value - returnn[i].detach()).pow(2).mean()
                #value_loss = (values[i] - img_Vs[i]).pow(2).mean()

                loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                # Update batch values

                update_entropy += entropy.item()
                update_value += value.mean().item()
                update_policy_loss += policy_loss.item()
                update_value_loss += value_loss.item()
                update_loss += loss

            # Update update values

            update_entropy /= (imagination - 1)
            update_value /= (imagination - 1)
            update_policy_loss /= (imagination - 1)
            update_value_loss /= (imagination - 1)
            update_loss /= (imagination - 1)

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

        elif img_method == 'imgdreamer':

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
            actions.append(dist.sample())
            rewards.append(self.acmodel.reward_decoder(
                torch.cat([states[-1], actions[-1].unsqueeze(-1)], dim=-1)).squeeze())
            #actions.append(exps.action[inds])
            #rewards.append(exps.reward[inds])

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


        elif img_method is None:
            # do not update through ppo algorithm

            log_entropies = [0]
            log_values = [0]
            log_policy_losses = [0]
            log_value_losses = [0]
            log_grad_norms = [0]

        else:
            raise NotImplementedError


        ################################################################
        # Representation of Dreamer
        ################################################################

        log_rep_losses = []
        log_recon_accs = []
        log_recon_losses = []
        log_recon_col_losses = []
        log_recon_obj_losses = []
        log_recon_state_losses = []
        log_reward_losses = []
        log_nonzero_reward_losses = []
        log_nonzero_reward_num = []
        log_zero_reward_losses = []
        log_zero_reward_num = []
        log_kl_losses = []

        # set recurrence = 1 when doesn't train representation modules on here

        if not 'rep' in self.loss_type:
            recurrence = 1
        else:
            recurrence = self.recurrence

        for inds in self._get_batches_starting_indexes(recurrence):
            # do not update through imaginary agent algorithm

            if (not 'rep' in self.loss_type) and (not 'img' in self.loss_type):

                log_rep_losses = [0]
                log_recon_accs = [0]
                log_recon_losses = [0]
                log_recon_col_losses = [0]
                log_recon_obj_losses = [0]
                log_recon_state_losses = [0]
                log_reward_losses = [0]
                log_nonzero_reward_losses = [0]
                log_nonzero_reward_num = [0]
                log_zero_reward_losses = [0]
                log_zero_reward_num = [0]
                log_kl_losses = [0]

                break

            # Initialize batch values

            update_rep_loss = 0
            update_recon_acc = 0
            update_recon_loss = 0
            update_recon_col_loss = 0
            update_recon_obj_loss = 0
            update_recon_state_loss = 0
            update_reward_loss = 0
            update_nonzero_reward_loss = 0
            update_nonzero_reward_num = 0
            update_zero_reward_loss = 0
            update_zero_reward_num = 0
            update_kl_loss = 0

            # Initialize memory

            if self.acmodel.recurrent:
                memory = exps.memory[inds]
                prev_state = exps.prev_state[inds]

            for i in range(recurrence):
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
                update_reward_loss += rep_loss['reward_loss']
                update_nonzero_reward_loss += rep_loss['nonzero_reward_loss']
                update_nonzero_reward_num += rep_loss['nonzero_reward_num']
                update_zero_reward_loss += rep_loss['zero_reward_loss']
                update_zero_reward_num += rep_loss['zero_reward_num']
                update_kl_loss += rep_loss['kl_loss'].item()

                # Update memories for next epoch

                if self.acmodel.recurrent and i < recurrence - 1:
                    if 'trxl' in self.mem_type:
                        memory = torch.stack(memory,dim=0).permute(2,0,1,3)
                    exps.memory[inds + i + 1] = memory.detach()
                    exps.prev_state[inds + i + 1] = state.detach()

            # Update update values

            update_rep_loss /= recurrence
            update_recon_acc /= recurrence
            update_recon_loss /= recurrence
            update_recon_col_loss /= recurrence
            update_recon_obj_loss /= recurrence
            update_recon_state_loss /= recurrence
            update_reward_loss /= recurrence
            update_nonzero_reward_loss /= recurrence
            #update_nonzero_reward_num /= recurrence
            update_zero_reward_loss /= recurrence
            #update_zero_reward_num /= recurrence
            update_kl_loss /= recurrence

            # Update representation modules

            if 'rep' in self.loss_type:
                self.rep_optimizer.zero_grad()
                update_rep_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.clip_dreamer)
                self.rep_optimizer.step()
            else:
                self.rep_optimizer.zero_grad()
                update_reward_loss.backward() # only training reward decoder for imaginary agent update
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.clip_dreamer)
                self.rep_optimizer.step()

            # Update log values

            log_rep_losses.append(update_rep_loss.item())
            log_recon_accs.append(update_recon_acc)
            log_recon_losses.append(update_recon_loss)
            log_recon_col_losses.append(update_recon_col_loss)
            log_recon_obj_losses.append(update_recon_obj_loss)
            log_recon_state_losses.append(update_recon_state_loss)
            log_reward_losses.append(update_reward_loss.item())
            log_nonzero_reward_losses.append(update_nonzero_reward_loss)
            log_nonzero_reward_num.append(update_nonzero_reward_num)
            log_zero_reward_losses.append(update_zero_reward_loss)
            log_zero_reward_num.append(update_zero_reward_num)
            log_kl_losses.append(update_kl_loss)

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
            "reward_loss": numpy.mean(log_reward_losses),
            "nonzero_reward_loss": numpy.mean(log_nonzero_reward_losses),
            "nonzero_reward_num": numpy.sum(log_nonzero_reward_num),
            "zero_reward_loss": numpy.mean(log_zero_reward_losses),
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
