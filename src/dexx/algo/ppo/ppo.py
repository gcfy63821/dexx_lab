# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

import os
import time
import torch

from dexx.algo.ppo.experience import ExperienceBuffer
from dexx.algo.models.models import ActorCritic, ActorCriticAsymmetric
from dexx.algo.models.running_mean_std import RunningMeanStd

from dexx.utils.misc import AverageScalarMeter

from tensorboardX import SummaryWriter


class PPO(object):
    def __init__(self, env, output_dir, full_config, create_output_dir=True):
        self.device = full_config.train["device"]
        self.network_config = full_config.train["network"]
        self.ppo_config = full_config.train["algorithm"]
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        action_space = self.env.action_space
        self.actions_num = action_space.shape[1]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = (self.observation_space.shape[1],)
        # ---- Priv Info ----
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.priv_info = self.ppo_config['priv_info']
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config["mlp"]["units"],
            'priv_mlp_units': self.network_config["priv_mlp"]["units"],
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': False,
            'priv_info_dim': self.priv_info_dim,
        }
        if self.network_config.get("asymmetric_ac", False):
            self.model = ActorCriticAsymmetric(net_config)
            print(f"INFO: USING ASYMMETRIC ACTOR-CRITIC (actor_obs={self.obs_shape[0]}, "
                f"critic_obs={self.obs_shape[0]+self.priv_info_dim})")
        else:
            self.model = ActorCritic(net_config)

        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
        self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
        if create_output_dir:
            os.makedirs(self.nn_dir, exist_ok=True)
            os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr, weight_decay=self.weight_decay)
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.kl_threshold = self.ppo_config['kl_threshold']
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        if create_output_dir:
            writer = SummaryWriter(self.tb_dif)
            self.writer = writer
        # ---- Wandb Logger ----
        self.use_wandb = False
        self.wandb = None


        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        # Episode success/failure over the last 100 COMPLETED episodes. Without
        # this the only training-time signal is reward, which can climb while the
        # task success rate does not move.
        self.episode_successes = AverageScalarMeter(100)
        self.episode_failures = AverageScalarMeter(100)
        # Conservative training proxy; unlike eval strict3, bad inits remain
        # in the denominator and success requires trajectory completion.
        self.episode_successes_strict = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size, self.obs_shape[0],
            self.actions_num, self.priv_info_dim, self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0
        # ---- Video Recording
        self.video_config = full_config.train.get("video", {"enabled": False})
        self.video_enabled = self.video_config.get("enabled", False)
        if self.video_enabled:
            self.video_folder = self.video_config.get("folder", os.path.join(self.output_dir, "videos"))
            self.video_interval = self.video_config.get("interval", 10000)
            self.video_length = self.video_config.get("length", 200)
            self.last_video_step = 0
            os.makedirs(self.video_folder, exist_ok=True)
            print(f"[INFO] Video recording enabled: folder={self.video_folder}, interval={self.video_interval}, length={self.video_length}")

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)

        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)

        # -------- wandb--------
        if self.use_wandb:
            log_dict = {
                "performance/RLTrainFPS":  self.agent_steps / self.rl_train_time,
                "performance/EnvStepFPS":  self.agent_steps / self.data_collect_time,
                "losses/actor_loss": torch.mean(torch.stack(a_losses)).item(),
                "losses/bounds_loss": torch.mean(torch.stack(b_losses)).item(),
                "losses/critic_loss": torch.mean(torch.stack(c_losses)).item(),
                "losses/entropy": torch.mean(torch.stack(entropies)).item(),
                "info/last_lr": self.last_lr,
                "info/e_clip": self.e_clip,
                "info/kl": torch.mean(torch.stack(kls)).item(),

            }
            data = self.storage.data_dict

            # ---------- Advantage / Value ----------
            advantages = data["advantages"]
            values = data["values"]
            returns = data["returns"]

            log_dict.update({
                "debug/adv_mean": advantages.mean().item(),
                "debug/adv_std": advantages.std().item(),
                "debug/adv_max": advantages.max().item(),

                "debug/value_mean": values.mean().item(),
                "debug/return_mean": returns.mean().item(),
                "debug/value_return_gap": (returns - values).abs().mean().item(),
            })

            # ---------- Policy behavior ----------
            actions = data["actions"]

            log_dict.update({
                "policy/action_abs_mean": actions.abs().mean().item(),
                "policy/action_abs_max": actions.abs().max().item(),
                "policy/action_clipped_ratio": (actions.abs() >= 0.999).float().mean().item(),
            })

            # ---------- Noise ----------
            sigmas = data["sigmas"]   # Gaussian policy std

            log_dict.update({
                "noise/action_std_mean": sigmas.mean().item(),
                "noise/action_std_min": sigmas.min().item(),
                "noise/action_std_max": sigmas.max().item(),
            })

            for k, v in self.extra_info.items():
                log_dict[f"extras/{k}"] = v.item() if torch.is_tensor(v) else v

            self.wandb.log(log_dict, step= self.agent_steps)

        for k, v in self.extra_info.items():
            self.writer.add_scalar(f'{k}', v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
            'priv_info': obs_dict['priv_info'],
        }
        res_dict = self.model.act(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.agent_steps = self.batch_size

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            
            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls)
            self.storage.data_dict = None


            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            # Episode success rate over completed episodes — the number to watch.
            mean_success = (self.episode_successes.get_mean()
                            if self.episode_successes.current_size > 0 else None)
            mean_failure = (self.episode_failures.get_mean()
                            if self.episode_failures.current_size > 0 else None)
            mean_success_strict = (self.episode_successes_strict.get_mean()
                                   if self.episode_successes_strict.current_size > 0 else None)
            if mean_success is not None:
                self.writer.add_scalar('success_rate/step', mean_success, self.agent_steps)
                self.writer.add_scalar('success_rate/iter', mean_success, self.epoch_num)
            if mean_success_strict is not None:
                self.writer.add_scalar('success_rate_strict/step', mean_success_strict, self.agent_steps)
                self.writer.add_scalar('success_rate_strict/iter', mean_success_strict, self.epoch_num)
            if mean_failure is not None:
                self.writer.add_scalar('failure_rate/step', mean_failure, self.agent_steps)
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, 'last'))

            if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                print(f'save current best reward: {mean_rewards:.2f}', flush=True)
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, 'best'))

            # Record video if enabled and interval reached
            if self.video_enabled and (self.agent_steps - self.last_video_step) >= self.video_interval:
                self._record_video()

            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                          f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                          f'Mean Rewards: {mean_rewards:.2f} | ' \
                          + (f'Success: {mean_success * 100:.1f}% | ' if mean_success is not None else '') \
                          + (f'Strict: {mean_success_strict * 100:.1f}% | ' if mean_success_strict is not None else '') \
                          + f'Current Best: {self.best_rewards:.2f}'
            print(info_string, flush=True)

        print('max steps achieved', flush=True)

    def _record_video(self):
        """Record a video of the current policy."""
        if not self.video_enabled:
            return
        
        try:
            import imageio
            import numpy as np
            
            print(f"[INFO] Recording video at agent step {self.agent_steps}")
            self.set_eval()
            
            # Reset environment and collect frames
            obs = self.env.reset()
            frames = []
            
            # Record video_length steps
            for step in range(self.video_length):
                # Get action from policy
                if isinstance(obs, dict):
                    processed_obs = self.running_mean_std(obs['obs'])
                    input_dict = {
                        'obs': processed_obs,
                        'priv_info': obs.get('priv_info', torch.zeros((self.num_actors, self.priv_info_dim), device=self.device)),
                    }
                else:
                    processed_obs = self.running_mean_std(obs)
                    input_dict = {'obs': processed_obs}
                
                res_dict = self.model.act(input_dict)
                actions = res_dict['actions']
                
                # Step environment
                obs, _, _, _ = self.env.step(actions)
                
                # Get frame from environment (if available)
                if hasattr(self.env, 'render'):
                    frame = self.env.render()
                    if frame is not None:
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        if len(frame.shape) == 4:  # [num_envs, H, W, C]
                            frame = frame[0]  # Take first environment
                        frames.append(frame)
                elif hasattr(self.env.unwrapped, 'render'):
                    frame = self.env.unwrapped.render()
                    if frame is not None:
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        if len(frame.shape) == 4:
                            frame = frame[0]
                        frames.append(frame)
            
            # Save video
            if len(frames) > 0:
                video_path = os.path.join(
                    self.video_folder,
                    f"video_step_{self.agent_steps:08d}.mp4"
                )
                # Convert frames to uint8 if needed
                frames_array = np.array(frames)
                if frames_array.dtype != np.uint8:
                    frames_array = (frames_array * 255).astype(np.uint8)
                
                imageio.mimwrite(video_path, frames_array, fps=30, codec='libx264', quality=8)
                print(f"[INFO] Video saved to: {video_path}")
                self.last_video_step = self.agent_steps
            else:
                print(f"[WARNING] No frames captured for video recording")
            
            self.set_train()
            
        except Exception as e:
            print(f"[WARNING] Failed to record video: {e}")
            import traceback
            traceback.print_exc()
            self.set_train()

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': obs_dict['priv_info'],
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs, priv_info = self.storage[i]

                obs = self.running_mean_std(obs)
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'priv_info': priv_info,
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']


                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    # Zero inside [-soft_bound, soft_bound], quadratic outside it.
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0
                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

                self.optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr
            kls.append(av_kls)

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls

    def play_steps(self):
        # Accumulate extras over the horizon for proper averaging
        extras_accum = {}
        extras_count = 0

        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = 0.01 * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            # Success is an EPISODE quantity: take it only from the envs that
            # just terminated, on the step they terminated. `succeeded_per_env`
            # is the pre-reset per-env flag from the env's `_get_rewards`.
            if done_indices.numel() > 0:
                _succ = infos.get('succeeded_per_env')
                if _succ is None and not getattr(self, '_warned_no_succ', False):
                    self._warned_no_succ = True
                    print("[PPO] WARNING: env does not publish 'succeeded_per_env'; "
                          "no success rate will be logged. The env's _get_rewards "
                          "must set it — note that subclasses override that method.",
                          flush=True)
                if _succ is not None:
                    self.episode_successes.update(_succ[done_indices])
                _fail = infos.get('failed_per_env')
                if _fail is not None:
                    self.episode_failures.update(_fail[done_indices])
                _strict = infos.get('succeeded_strict_per_env')
                if _strict is not None:
                    self.episode_successes_strict.update(_strict[done_indices])

            assert isinstance(infos, dict), 'Info Should be a Dict'
            for k, v in infos.items():
                # only accumulate scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    val = v.item() if torch.is_tensor(v) else v
                    if k not in extras_accum:
                        extras_accum[k] = 0.0
                    extras_accum[k] += val
            extras_count += 1

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        # Average accumulated extras over the horizon
        self.extra_info = {}
        if extras_count > 0:
            for k, v in extras_accum.items():
                self.extra_info[k] = v / extras_count

        # Debug: print extras summary once every 10 epochs
        if not hasattr(self, '_extras_debug_counter'):
            self._extras_debug_counter = 0
        self._extras_debug_counter += 1
        if self._extras_debug_counter % 10 == 1:
            print(f"[EXTRAS DEBUG] horizon={extras_count}, dones_total={self.dones.sum().item():.0f}")
            for k in sorted(self.extra_info.keys()):
                print(f"  {k}: {self.extra_info[k]:.6f}")
            # Also print what keys are in infos but NOT scalar (filtered out)
            non_scalar_keys = [k for k, v in infos.items()
                               if not (isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0))]
            if non_scalar_keys:
                print(f"  [non-scalar keys filtered out]: {non_scalar_keys}")

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps += self.batch_size
        self.storage.computer_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
