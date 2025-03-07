# advanced_ppo_trainer.py
import copy
from collections import namedtuple

import numpy as np
import torch
from tqdm import trange

from log_utils import setupLogging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RolloutStep = namedtuple("RolloutStep", ["state", "action", "reward", "value", "log_prob", "done"])

class AdvancedPPOTrainer:
    def __init__(
        self,
        env,
        policy,
        baseline,
        quant_types,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=1e-3,
        kl_coeff=0.5,
        entropy_coeff=0.1,
        train_policy_iters=3,
        results_path="results"):
        """
        Args:
          env: EnhancedQuantizationEnv
          policy: PolicyNetPPO
          baseline: BaselineNetwork
          quant_types: list of discrete actions
          gamma: discount factor
          lam: GAE-lambda or standard advantage calculation factor
          clip_ratio: PPO clip hyperparam (epsilon)
          train_policy_iters: how many epochs of policy update per episode
        """

        self.kl_coeff = kl_coeff
        self.logger = setupLogging(results_path)
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_policy_iters = train_policy_iters
        self.quant_types = quant_types
        self.entropy_coeff = entropy_coeff

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def run_episode(self):
        """
        Run one entire episode (which in your environment means quantizing
        all layers in sequence). Collect transitions for PPO.
        Returns:
          rollout: list of RolloutStep tuples for each step
          ep_info: dict from env (like final_loss, bits chosen, etc.)
        """
        state = self.env.reset()  # initial state
        done = False
        rollout = []

        while not done:
            # Convert state to torch
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)

            # Get distribution, sample action
            dist = self.policy.action_distribution(state_t)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t)

            # Convert to python scalar
            action_idx = action_t.item()
            log_prob = log_prob_t.item()
            # actual bits from bit_options
            action_quant_type = self.quant_types[action_idx]

            # Baseline value
            value_t = self.baseline.forward(state_t).item()

            # Step env
            next_state, reward, done, info = self.env.step(action_quant_type)

            # Store in rollout
            rollout.append(
                RolloutStep(
                    state=state,
                    action=action_idx,
                    reward=reward,
                    value=value_t,
                    log_prob=log_prob,
                    done=done
                )
            )

            state = next_state

        return rollout, info

    def compute_gae(self, rollout):
        """
        Computes GAE-lambda advantages and returns.
        We assume the episode ends fully, so for the last step we set next_value=0.
        If done[t], we do not bootstrap further from that step.
        """
        rewards = [step.reward for step in rollout]
        values = [step.value for step in rollout]
        dones = [step.done for step in rollout]
        length = len(rollout)

        advantages = np.zeros(length, dtype=np.float32)
        gae = 0.0

        # Go backwards from last step to first
        for t in reversed(range(length)):
            if t == length - 1:
                next_value = 0.0
                next_nonterminal = 1.0 - float(dones[t])
            else:
                next_value = values[t+1] if not dones[t] else 0.0
                next_nonterminal = 1.0 - float(dones[t])

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lam * gae * next_nonterminal
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)
        return returns, advantages


    def compute_returns_and_advantages_gae(self, rollout):
        return self.compute_gae(rollout)


    def compute_returns_and_advantages_simple(self, rollout):
        """
        Standard returns + advantage = returns - baseline_value.
        """
        rewards = [step.reward for step in rollout]

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # Compute advantage as A_t = R_t - V(s_t)
        advantages = []
        for t in range(len(rollout)):
            advantages.append(returns[t] - rollout[t].value)

        return returns, advantages

    def update_baseline(self, rollout, returns):
        """
        Fit baseline (value function) using MSE.
        """
        states = np.array([step.state for step in rollout], dtype=np.float32)
        targets = np.array(returns, dtype=np.float32)
        loss = self.baseline.update(states, targets)
        return loss

    def update_policy_ppo(self, rollout, advantages):
        """
        Perform the clipped PPO update. We'll do a few epochs over the entire
        collected batch (which is the entire episode).
        """
        states_np = np.array([step.state for step in rollout], dtype=np.float32)
        actions_np = np.array([step.action for step in rollout], dtype=np.int64)
        old_log_probs_np = np.array([step.log_prob for step in rollout], dtype=np.float32)
        adv_np = np.array(advantages, dtype=np.float32)

        states = torch.from_numpy(states_np).to(device)
        actions = torch.from_numpy(actions_np).to(device)
        old_log_probs = torch.from_numpy(old_log_probs_np).to(device)
        adv_t = torch.from_numpy(adv_np).to(device)

        # Normalize advantages if desired
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # old policy
        old_policy = copy.deepcopy(self.policy)
        old_policy.eval()

        # Multiple epochs (train_policy_iters)
        for _ in range(self.train_policy_iters):
            dist = self.policy.action_distribution(states)
            # the old distribution for KL
            with torch.no_grad():
                old_dist = old_policy.action_distribution(states)

            # PPO objectives
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            obj1 = ratio * adv_t
            obj2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t

            policy_loss = -torch.mean(torch.min(obj1, obj2))

            # KL divergence new vs old
            kl_div = torch.distributions.kl_divergence(old_dist, dist).mean()
            # Entropy bonus
            entropy = dist.entropy().mean()
            policy_loss = policy_loss + self.kl_coeff * kl_div - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

        return policy_loss.item()

    def train(self, num_episodes=10):
        """
        Main training loop:
         1) For each episode, run full rollout
         2) Compute returns + advantages
         3) Update baseline
         4) Update policy with PPO
         5) Log results
        """
        all_rewards = []
        infos = []
        episode_iter = trange(num_episodes, desc="PPO Training")
        for ep in episode_iter:
            rollout, info = self.run_episode()

            ep_info = info.get("episode_summary", {})

            # final_loss, memory_saved, layer_bits, ...
            final_loss = ep_info.get("final_loss", None)
            chosen_qtypes = ep_info.get("layer_bits", None)

            # 1) Returns + advantages
            returns, advantages = self.compute_returns_and_advantages_gae(rollout)

            # 2) Baseline update
            baseline_loss = self.update_baseline(rollout, returns)

            # 3) Policy update
            policy_loss = self.update_policy_ppo(rollout, advantages)

            # sum of the raw rewards from the rollout
            ep_reward = sum([step.reward for step in rollout])
            all_rewards.append(ep_reward)

            ep_info["reward"] = ep_reward
            ep_info["policy_loss"] = policy_loss
            ep_info["baseline_loss"] = baseline_loss

            infos.append(info)

            self.logger.info(f"Episode {ep}, reward={ep_reward:.4f}, final_loss={final_loss}, policy_loss={policy_loss:.6f}, baseline_loss={baseline_loss:.6f}, qtypes={chosen_qtypes}")
            episode_iter.set_postfix({
                "ep_reward": f"{ep_reward:.4f}",
                "policy_loss": f"{policy_loss:.4f}",
            })

        return all_rewards, infos
