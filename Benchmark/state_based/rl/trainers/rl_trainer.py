"""
训练的主要代码
"""
import copy
import os
import numpy as np
import gym
import torch
import pybullet as p
from mpi4py import MPI

from agents.factory import make_agent
from components.checkpointer import CheckpointHandler
from components.logger import Logger, WandBLogger, logger
from modules.replay_buffer import HerReplayBuffer, get_buffer_sampler
from modules.samplers import Sampler, RLIFSampler
from utils.general_utils import (AverageMeter, Every, Timer, Until,
                                   set_seed_everywhere)
from utils.mpi import (mpi_gather_experience_episode,
                         mpi_gather_experience_rollots, mpi_sum,
                         update_mpi_config)
from utils.rl_utils import RolloutStorage, get_env_params, init_buffer
from .base_trainer import BaseTrainer


class RLTrainer(BaseTrainer):
    def _setup(self):
        self._setup_env()       # Environment
        self._setup_buffer()    # Relay buffer
        self._setup_agent()     # Agent
        self._setup_sampler()   # Sampler
        self._setup_logger()    # Logger
        self._setup_misc()      # MISC

        if self.is_chef:
            self.termlog.info('Setup done')

    def _setup_env(self):
        self.train_env = gym.make(self.cfg.task)
        # self.eval_env = gym.make(self.cfg.task)
        self.env_params = get_env_params(self.train_env, self.cfg)
        
    def _setup_buffer(self):
        self.buffer_sampler = get_buffer_sampler(self.train_env, self.cfg.agent.sampler)
        self.buffer = HerReplayBuffer(buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params,
                            batch_size=self.cfg.batch_size, sampler=self.buffer_sampler)
        if self.cfg.get('use_demo'):
            self.demo_buffer = HerReplayBuffer(buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params,
                                batch_size=self.cfg.batch_size, sampler=self.buffer_sampler)
        else:
            self.demo_buffer = None

    def _setup_agent(self):
        self.agent = make_agent(self.env_params, self.buffer_sampler, self.cfg.agent)

    def _setup_sampler(self):
        self.train_sampler = Sampler(self.train_env, self.agent, self.env_params['max_timesteps'])
        self.eval_sampler = Sampler(self.train_env, self.agent, self.env_params['max_timesteps'])

    def _setup_logger(self):
        update_mpi_config(self.cfg)
        if self.is_chef:
            exp_name = f"{self.cfg.task}_{self.cfg.agent.name}_demo{self.cfg.num_demo}_seed{self.cfg.seed}"
            if self.cfg.postfix is not None:
                exp_name =  exp_name + '_' + str(self.cfg.postfix) 
            if self.cfg.use_wb:
                resume_option = self.cfg.get('resume', False)
                self.wb = WandBLogger(exp_name=exp_name, project_name=self.cfg.project_name, entity=self.cfg.entity_name, \
                        path=self.work_dir, conf=self.cfg, resume=resume_option)
            self.logger = Logger(self.work_dir)
            self.termlog = logger
        else: 
            self.wb, self.logger, self.termlog = None, None, None
    
    def _setup_misc(self):
        # init_buffer(self.cfg, self.buffer, self.agent, normalize=False) # important for awac, amp
        # init_buffer(self.cfg, self.demo_buffer, self.agent, normalize=True)
        if self.cfg.get('use_demo'):
            if self.is_chef:
                self.termlog.info(f"Demo path found: {self.cfg.demo_path}. Loading demonstrations...")
                
            # 只有在提供了 demo_path 时，才调用 init_buffer
            init_buffer(self.cfg, self.buffer, self.agent, normalize=False)
            init_buffer(self.cfg, self.demo_buffer, self.agent, normalize=True)
        else:
            if self.is_chef:
                self.termlog.warning("No demo_path configured. Skipping demonstration loading. Running in pure RL mode.")

        if self.is_chef:
            self.model_dir = self.work_dir / 'model'
            self.model_dir.mkdir(exist_ok=True)

            self.buffer_dir = self.work_dir / 'buffer'
            self.buffer_dir.mkdir(exist_ok=True)

            if self.cfg.get('clean_model_dir', False): # 使用 .get() 避免没设置时报错
                self.termlog.info(f"Cleaning model & buffer directory: {self.model_dir}")
                for file in os.listdir(self.model_dir):
                    os.remove(self.model_dir / file)
                for file in os.listdir(self.buffer_dir):
                    os.remove(self.buffer_dir / file)
            # for file in os.listdir(self.model_dir):
            #     os.remove(self.model_dir / file)

        self.device = torch.device(self.cfg.device)
        self.timer = Timer()
        self._global_step = 0
        self._global_episode = 0
        set_seed_everywhere(self.cfg.seed)
    
    def train(self):
        n_train_episodes = int(self.cfg.n_train_steps / self.env_params['max_timesteps'])
        # n_eval_episodes = int(n_train_episodes / self.cfg.n_eval) * self.cfg.mpi.num_workers
        # n_save_episodes = int(n_train_episodes / self.cfg.n_save) * self.cfg.mpi.num_workers
        # n_log_episodes = int(n_train_episodes / self.cfg.n_log) * self.cfg.mpi.num_workers

        n_eval_episodes = int(self.cfg.eval_interval * self.cfg.mpi.num_workers)
        n_save_episodes = int(self.cfg.save_interval * self.cfg.mpi.num_workers)
        n_log_episodes = int(self.cfg.log_interval * self.cfg.mpi.num_workers)

        # save 间隔必须大于 eval 间隔, 而且要规整到 eval 的整数倍
        assert n_save_episodes > n_eval_episodes
        if n_save_episodes % n_eval_episodes != 0:
            n_save_episodes = int(n_save_episodes / n_eval_episodes) * n_eval_episodes

        train_until_episode = Until(n_train_episodes)
        save_every_episodes = Every(n_save_episodes)
        eval_every_episodes = Every(n_eval_episodes)
        log_every_episodes = Every(n_log_episodes)
        seed_until_steps = Until(self.cfg.n_seed_steps)

        if self.is_chef:
            self.termlog.info('Starting training')
        while train_until_episode(self.global_episode):
            self._train_episode(log_every_episodes, seed_until_steps)

            if eval_every_episodes(self.global_episode):
                score, reward = self.eval()

            if not self.cfg.dont_save and save_every_episodes(self.global_episode) and self.is_chef:
                filename =  CheckpointHandler.get_ckpt_name(self.global_episode)
                # TODO(tao): expose scoring metric
                CheckpointHandler.save_checkpoint({
                    'episode': self.global_episode,
                    'global_step': self.global_step,
                    'state_dict': self.agent.state_dict(),
                    'o_norm': self.agent.o_norm,
                    'g_norm': self.agent.g_norm,
                    'score': score,
                }, self.model_dir, filename)
                self.termlog.info(f'Save checkpoint to {os.path.join(self.model_dir, filename)}')

                # --- 调用 buffer 自带的 save 方法 ---
                # self.termlog.info("Saving replay buffer...")
                self.buffer.save(save_dir=self.buffer_dir, episode=self.global_episode)
                self.termlog.info(f"Save Replay buffer to {os.path.join(self.buffer_dir, f'replay_buffer_ep{self.global_episode}.zip')}")

    def resume_train(self):
        """
        从一个 checkpoint (包括模型和Replay Buffer) 恢复训练.
        """
        if self.is_chef:
            # --- 加载模型 Checkpoint 并恢复状态 ---
            self.termlog.info(f"Resuming training from checkpoint specified by ckpt_episode={self.cfg.ckpt_episode}...")
            checkpoint = CheckpointHandler.load_checkpoint(
                self.model_dir, self.agent, self.device, self.cfg.ckpt_episode
            )
            
            self._global_episode = checkpoint.get('episode', 0)
            self._global_step = checkpoint.get('global_step', 0)
            self.termlog.info(f"Resumed model state from episode {self.global_episode}, global step {self.global_step}")

            # --- 加载 Replay Buffer ---
            # self.termlog.info("Attempting to load corresponding replay buffer...")
            buffer_file = os.path.join(self.buffer_dir, f"replay_buffer_ep{self._global_episode}.zip")
            meta_file = os.path.join(self.buffer_dir, f"idx_size_ep{self._global_episode}.npy")

            if os.path.exists(buffer_file) and os.path.exists(meta_file):
                self.buffer.load(save_dir=self.buffer_dir, episode=self._global_episode)
                self.termlog.info(f"Replay buffer for episode {self._global_episode} successfully loaded.")
            else:
                self.termlog.warning(f"Warning: Replay buffer for episode {self._global_episode} not found.")
                self.termlog.warning("Starting with an empty replay buffer. This may affect performance.")

            # self.termlog.info("Starting resumed training loop...")
        self.sync_state()  # 同步所有 worker 的状态
        self.train()

    def _train_episode(self, log_every_episodes, seed_until_steps):
        # sync network parameters across workers
        if self.use_multiple_workers > 1:
            self.agent.sync_networks()

        self.timer.reset()
        batch_time = AverageMeter()
        ep_start_step = self.global_step
        metrics = None

        # collect experience, multi-worker
        rollout_storage = RolloutStorage()
        # sample an episode
        episode, rollouts, env_steps = self.train_sampler.sample_episode(is_train=True, render=False, random_act=seed_until_steps(ep_start_step))
        if self.use_multiple_workers:
            transitions_batch = mpi_gather_experience_episode(rollouts)
        else:
            transitions_batch = rollouts

        # update status
        rollout_storage.append(episode)
        rollout_status = rollout_storage.rollout_stats()
        self._global_step += int(mpi_sum(env_steps))
        self._global_episode += int(mpi_sum(1))

        # save to buffer
        self.buffer.store_episode(copy.deepcopy(transitions_batch))
        self.agent.update_normalizer(copy.deepcopy(transitions_batch))

        # update policy, only if not in seed mode (非随机采样阶段)
        if not seed_until_steps(ep_start_step):
            if self.is_chef:
                metrics = self.agent.update(self.buffer, self.demo_buffer)
            # 多worker时, 要各网络的参数保持同步(广播)
            if self.use_multiple_workers:
                self.agent.sync_networks()
        else:
            self.connect_EGL()  # 定时与 EGL 渲染器保持连接

        # log results
        if log_every_episodes(self.global_episode) and self.is_chef:
            if metrics is not None:
                elapsed_time, total_time = self.timer.reset()
                batch_time.update(elapsed_time)
                togo_train_time = batch_time.avg * (self.cfg.n_train_steps - ep_start_step) / env_steps

                self.logger.log_metrics(metrics, self.global_step, ty='train')      # 在此输出训练结果
                with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                    log('fps', env_steps / elapsed_time)
                    log('total_time', total_time)
                    log('episode_reward', rollout_status.avg_reward)
                    log('episode_length', env_steps)
                    log('episode_sr', rollout_status.avg_success_rate)
                    log('episode', self.global_episode)
                    log('step', self.global_step)
                    log('ETA', togo_train_time)
                if self.cfg.use_wb:
                    self.wb.log_outputs(metrics, None, log_images=False, step=self.global_step, is_train=True)

            else:
                self.termlog.info(
                    f"--- Seeding Phase --- | "
                    f"Episode: {self.global_episode} | "
                    f"Step: {self.global_step}/{self.cfg.n_seed_steps} | "
                    f"Avg Reward: {rollout_status.avg_reward:.2f}"
                )

    def eval(self):
        '''Eval agent.'''
        eval_rollout_storage = RolloutStorage()     # 会存放每一个episode的结果(image, reward, info, success)
        for _ in range(self.cfg.n_eval_episodes):
            # 与环境交互采样, 得到结果
            episode, _, env_steps = self.eval_sampler.sample_episode(is_train=False, render=True)
            eval_rollout_storage.append(episode)
        rollout_status = eval_rollout_storage.rollout_stats()
        if self.use_multiple_workers:
            rollout_status = mpi_gather_experience_rollots(rollout_status)
            for key, value in rollout_status.items():
                rollout_status[key] = value.mean()

        if self.is_chef:
            if self.cfg.use_wb:
                self.wb.log_outputs(rollout_status, eval_rollout_storage, log_images=True, step=self.global_step)
            with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode', self.global_episode)
                log('step', self.global_step)

        del eval_rollout_storage
        return rollout_status.avg_success_rate, rollout_status.avg_reward

    def eval_ckpt(self):
        '''Eval checkpoint.'''
        if self.is_chef:
            CheckpointHandler.load_checkpoint(
                self.cfg.ckpt_dir, self.agent, self.device, self.cfg.ckpt_episode
            )
            avg_success_rate, avg_reward = self.eval()
            self.termlog.info(f'Successful rate: {avg_success_rate}; Average reward: {avg_reward}')

    def sync_state(self):
        """Sync training state across workers."""
        if self.is_chef:
            state = np.array([self._global_episode, self._global_step], dtype='int64')
        else:
            state = np.empty(2, dtype='int64')

        # --- 进行状态广播 (所有进程都会执行) ---
        comm = MPI.COMM_WORLD
        comm.Bcast(state, root=0) # Chef (root=0) 发送 state，Workers 接收并覆盖自己的 state

        # --- 所有进程用同步后的状态更新自己 ---
        self._global_episode = state[0]
        self._global_step = state[1]
        
        if self.is_chef:
            self.termlog.info(f"All workers' state synchronized.")

    def connect_EGL(self, heartbeat_interval_steps=200):
        if self.use_multiple_workers:
            # 我们可以用心跳间隔来控制频率，避免每一步都执行
            # 比如每 200 个全局步骤执行一次
            if self.global_episode % heartbeat_interval_steps == 0:
                try:
                    # 这是一个非常轻量的调用，它只是“碰一下”渲染器，但不会做任何耗时的工作
                    p.getCameraImage(width=1, height=1)
                except Exception:
                    # 忽略这里的任何潜在错误，我们的目的只是为了保持连接
                    pass

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def is_chef(self):
        return self.cfg.mpi.is_chef

    @property
    def use_multiple_workers(self):
        return self.cfg.mpi.num_workers > 1

class RLIFRLTrainer(RLTrainer):
    def _setup_sampler(self):
        self.train_sampler = RLIFSampler(self.train_env, self.agent, self.env_params['max_timesteps'])
        self.eval_sampler = Sampler(self.train_env, self.agent, self.env_params['max_timesteps'])

    def train(self):
        n_train_episodes = int(self.cfg.n_train_steps / self.env_params['max_timesteps'])
        n_eval_episodes = int(n_train_episodes / self.cfg.n_eval) * self.cfg.mpi.num_workers
        n_save_episodes = int(n_train_episodes / self.cfg.n_save) * self.cfg.mpi.num_workers
        n_log_episodes = int(n_train_episodes / self.cfg.n_log) * self.cfg.mpi.num_workers

        assert n_save_episodes > n_eval_episodes
        if n_save_episodes % n_eval_episodes != 0:
            n_save_episodes = int(n_save_episodes / n_eval_episodes) * n_eval_episodes

        train_until_episode = Until(n_train_episodes)
        save_every_episodes = Every(n_save_episodes)
        eval_every_episodes = Every(n_eval_episodes)
        log_every_episodes = Every(n_log_episodes)
        seed_until_steps = Until(self.cfg.n_seed_steps)

        if self.is_chef:
            self.termlog.info('Starting training')
        while train_until_episode(self.global_episode):

            self._train_episode(log_every_episodes, seed_until_steps)

            if eval_every_episodes(self.global_episode):
                score = self.eval()

            if not self.cfg.dont_save and save_every_episodes(self.global_episode) and self.is_chef:
                filename =  CheckpointHandler.get_ckpt_name(self.global_episode)
                # TODO(tao): expose scoring metric
                CheckpointHandler.save_checkpoint({
                    'episode': self.global_episode,
                    'global_step': self.global_step,
                    'state_dict': self.agent.state_dict(),
                    'o_norm': self.agent.o_norm,
                    'g_norm': self.agent.g_norm,
                    'score': score,
                }, self.model_dir, filename)
                self.termlog.info(f'Save checkpoint to {os.path.join(self.model_dir, filename)}')

    def eval(self):
        '''Eval agent.'''
        eval_rollout_storage = RolloutStorage()
        for _ in range(self.cfg.n_eval_episodes):
            episode, _, env_steps = self.eval_sampler.sample_episode(is_train=False, render=True)
            eval_rollout_storage.append(episode)
        rollout_status = eval_rollout_storage.rollout_stats()
        if self.use_multiple_workers:
            rollout_status = mpi_gather_experience_rollots(rollout_status)
            for key, value in rollout_status.items():
                rollout_status[key] = value.mean()

        if self.is_chef:
            if self.cfg.use_wb:
                self.wb.log_outputs(rollout_status, eval_rollout_storage, log_images=True, step=self.global_step)
            with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode', self.global_episode)
                log('step', self.global_step)

        del eval_rollout_storage
        return rollout_status.avg_success_rate
        
