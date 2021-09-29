from multiprocessing import Process, Pipe
import gym

import numpy as np
from gym_minigrid.minigrid import Grid
from PIL import Image

def get_agentview(img, agent_view_size, img_encode, unity_env=False):
    if unity_env:
        img = Image.fromarray(img)
        img = np.asarray(img.resize((64, 64), Image.BILINEAR), dtype=np.float) / 255.0
        return img
    else:
        if img_encode:
            grid, vis_mask = Grid.decode(img)
            partialview = grid.render(
                8,
                agent_pos=(agent_view_size // 2, agent_view_size - 1),
                agent_dir=3,
                highlight_mask=vis_mask
            )
            partialview = Image.fromarray(partialview)
            partialview = np.asarray(partialview.resize((64, 64), Image.BILINEAR), dtype=np.float) / 255.0
            return partialview
        else:
            return img

def worker(conn, env, img_encode, unity_env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            if unity_env:
                img = get_agentview(obs[0], None, None, unity_env)
                obs = {}
                obs['image'] = img
            else:
                obs = env.gen_obs()
                obs['image'] = get_agentview(obs['image'], env.agent_view_size, img_encode)
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            if unity_env:
                img = get_agentview(obs[0], None, None, unity_env)
                obs = {}
                obs['image'] = img
            else:
                obs = env.gen_obs()
                obs['image'] = get_agentview(obs['image'], env.agent_view_size, img_encode)
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, img_encode, unity_env):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.img_encode = img_encode
        self.unity_env = unity_env

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, img_encode, unity_env))
            p.daemon = True
            p.start()
            remote.close()

    def env0_reset(self):
        obs = self.envs[0].reset()
        if self.unity_env:
            img = get_agentview(obs[0], None, None, self.unity_env)
            obs = {}
            obs['image'] = img
        else:
            obs = self.envs[0].gen_obs()
            obs['image'] = get_agentview(obs['image'], self.envs[0].agent_view_size, self.img_encode)
        return obs

    def env0_step(self, action):
        obs, reward, done, info = self.envs[0].step(action)
        if self.unity_env:
            img = get_agentview(obs[0], None, None, self.unity_env)
            obs = {}
            obs['image'] = img
        else:
            obs = self.envs[0].gen_obs()
            obs['image'] = get_agentview(obs['image'], self.envs[0].agent_view_size, self.img_encode)
        return (obs, reward, done, info)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.env0_reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.env0_step(actions[0])
        if done:
            obs = self.env0_reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError
