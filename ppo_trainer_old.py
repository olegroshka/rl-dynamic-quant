# ppo_trainer_old.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import trange  # or tqdm, whichever you prefer

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

def ppo_train(env, policy, num_episodes=10, gamma=0.99, epsilon=0.2, lr=1e-3, bit_options=None, ppo_updates_per_iteration=3, dev=device):

    if bit_options is None:
        bit_options = [4, 8, 8, 16]
    #bit_options = [4, 8, 8, 16]
    #bit_options = [2, 4, 8, 16]
    #bit_options = [2, 4, 8]

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    policy.to(dev)
    policy.train()

    final_info = {}

    episode_iter = trange(num_episodes, desc="Training PPO")
    for episode in range(episode_iter):
        state = env.reset()
        done = False

        states = []
        actions = []
        rewards = []
        old_log_probs = []
        episode_rewards = []

        # 1) Collect a full episode
        while not done:

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(dev)
            logits = policy(state_tensor)
            action_dist = F.softmax(logits, dim=-1)

            action_idx = torch.multinomial(action_dist, 1).item()
            action_bits = bit_options[action_idx]

            # Detach old log-prob so it doesn't keep the graph
            log_prob = torch.log(action_dist[0, action_idx]).detach()

            next_state, reward, done, info = env.step(action_bits)

            states.append(state_tensor.detach())  # or .cpu()
            actions.append(action_idx)
            rewards.append(reward)
            old_log_probs.append(log_prob)
            episode_rewards.append(reward)

            state = next_state

        if done and "chosen_bits_for_layer" in info:
            final_info = info
            logger.info(f"[Episode {episode}]: final layer bit choices = {info['chosen_bits_for_layer']}")


        # 2) Compute returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        states = torch.cat(states, dim=0)
        actions = torch.LongTensor(actions).to(dev)
        # stack and detach the old_log_probs as well
        old_log_probs = torch.stack(old_log_probs).to(dev)
        returns = torch.FloatTensor(returns).to(dev)

        # 3) Multiple PPO updates
        for _ in range(ppo_updates_per_iteration):
            # New forward pass to get current log probs
            logits = policy(states)
            new_action_dists = F.softmax(logits, dim=-1)
            new_log_probs = torch.log(new_action_dists.gather(1, actions.unsqueeze(1)).squeeze(1))

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            policy_loss = -torch.min(ratio * returns, clipped_ratio * returns).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        total_reward = sum(rewards)
        logger.info(f"Episode {episode}/{num_episodes - 1}: total_reward={total_reward:.4f}, final_loss={policy_loss.item():.4f}")
        episode_iter.set_postfix({
            "episode_reward": f"{total_reward:.2f}",
        })

    return final_info
