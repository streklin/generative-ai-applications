#!/usr/bin/env python
# coding: utf-8

# # Generative AI Applications Project
# * **Gene Foxwell**
# 
# ## Overview
# 
# This project looks at how a Generative Pretrained Transformer Model (GPT) can be used to solve Reinforcement Learning Problems. The idea is based on the Decision Transformers paper (PAPER). We will use the Human Relocate Data from Minari's DR4RL section. Further information on the Relocate data can be found here (PAPER). The relocate problem was originally introduced here (PAPER). 
# 
# The basic problem is to train virtual 24 DoF robotic hand to pick up a ball from one location and move it to another.
# 
# We'll attack this problem in the following sequence:
# 
# * Build a GPT Model that we can use with a Decision Transformer.
# * We'll build an implementation of the Decision Transformer Algorithm.
# * We'll train the algorithm on the Human generated data for the Relocate the problem.
# * Summary of results.
# 
# Let's get started...

# ## Transformer Model
# 
# We'll build our transformer model based on the architecture described in (ATTENTION PAPER) as well as examples from Udacity's course work.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from collections import deque

import minari
import gymnasium as gym
import gymnasium_robotics

import imageio
from IPython.display import Video


# In[ ]:


# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(
            config["embed_dim"], config["head_size"], config["use_bias"]
        )
        self.K_weights = nn.Linear(
            config["embed_dim"], config["head_size"], config["use_bias"]
        )
        self.V_weights = nn.Linear(
            config["embed_dim"], config["head_size"], config["use_bias"]
        )

        self.dropout = nn.Dropout(config["dropout_rate"])

        casual_attention_mask = torch.tril(
            torch.ones(config["context_size"], config["context_size"])
        )
        self.register_buffer("casual_attention_mask", casual_attention_mask)

    def forward(self, input):  # (B, C, embedding_dim)
        batch_size, tokens_num, embedding_dim = input.shape
        Q = self.Q_weights(input)  # (B, C, head_size)
        K = self.K_weights(input)  # (B, C, head_size)
        V = self.V_weights(input)  # (B, C, head_size)

        # Matrix Multiplay Q x K transpose to get the dot product of the query vectors with the key vectors
        attention_scores = Q @ K.transpose(1, 2)  # (B, C, C)

        # scale attention scores, scalled by square root of the dimensionality of the key vectors
        attention_scores = attention_scores / (K.shape[-1] ** 0.5)

        # mask the attention scores
        attention_scores = attention_scores.masked_fill(
            self.casual_attention_mask[:tokens_num, :tokens_num] == 0, -torch.inf
        )

        # calculate softmax values
        attention_scores = torch.softmax(attention_scores, dim=-1)

        # apply dropout for regularization
        attention_scores = self.dropout(attention_scores)

        # multiply attention scores by the value function.
        return attention_scores @ V  # (B, C, head_size)


# In[ ]:


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # initialize the individual AttentionHead objects
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)

        # Feedforward connection for after the attention heads
        self.linear = nn.Linear(config["embed_dim"], config["embed_dim"])

        # Dropout regularization.
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input):
        # execute heads in ||
        heads_outputs = [head(input) for head in self.heads]

        # concatenate the outputs into a single tensor
        scores_change = torch.cat(heads_outputs, dim=-1)

        # run the results through a feed forward network.
        scores_change = self.linear(scores_change)

        # regularization and return results
        return self.dropout(scores_change)


# In[ ]:


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.linear_layers = nn.Sequential(
            nn.Linear(config["embed_dim"], config["embed_dim"] * 4),
            nn.GELU(),
            nn.Linear(config["embed_dim"] * 4, config["embed_dim"]),
            nn.Dropout(config["dropout_rate"]),
        )

    def forward(self, input):
        return self.linear_layers(input)


# In[ ]:


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["embed_dim"])

        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["embed_dim"])

    def forward(self, input):
        residual = input
        x = self.multi_head(self.layer_norm_1(input))
        x = x + residual

        residual = x
        x = self.feed_forward(self.layer_norm_2(x))
        return x + residual


# In[ ]:


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = [Block(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(config["embed_dim"])


    def forward(self, input_embeddings):
        """
        Forward step for the transformer model. The DecisionTransformer already handles embeddings and positional encodings.
        We are simply making predictions using the AttentionHeads and returning the final hidden layer.
        """

        # Pass the embeddings through the stacked Transformer blocks
        x = self.layers(input_embeddings)

        # Apply the final layer normalization
        return self.layer_norm(x)


# ## Decision Transformer
# 
# We'll base our implementation of the Decision Transformer based on the original papers github repo found here: (https://github.com/kzl/decision-transformer)

# In[ ]:


class DecisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # size of the hidden layer
        self.hidden_size = config["embed_dim"]

        # maximum number of timetimes in an episode
        max_ep_length = config["max_ep_length"]

        # dimension of the state space
        self.state_dim = config["state_dim"]

        # dimension of the action space.
        self.action_dim = config["action_dim"]

        # embedding layers for timestamps, returns, states, and actions (t,r,s,a)
        # remember, at each time timestamp, we have a triple:
        #   r = expected return
        #   s = state
        #   a = action
        self.embed_timestep = nn.Embedding(max_ep_length, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(self.action_dim, self.hidden_size)

        # Normalization of the embedding layers
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # action prediction layer
        self.predict_action = nn.Linear(self.hidden_size, self.action_dim)

        # create an instance of the transformer model
        # this will be used to generate the next steps in the sequence.
        self.transformer = TransformerModel(config)

    def forward(self, states, actions, returns_to_go, timesteps):
        """
        Generate the next action using the previous Returns, States, Actions, and Timestamps.
        Based on the DecisionTransformer algorithm
        """
        batch_size, seq_length = states.shape[0], states.shape[1]

        pos_embedding = self.embed_timestep(timesteps)

        state_embeddings = self.embed_state(states) + pos_embedding
        action_embeddings = self.embed_action(actions) + pos_embedding
        returns_embeddings = self.embed_return(returns_to_go) + pos_embedding

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)

        input_embeddings = self.embed_ln(stacked_inputs)

        hidden_states = self.transformer(input_embeddings)

        # we are making predictions for a continious system with actions between [-1, 1]
        return torch.tanh(self.predict_action(hidden_states[:, 1::3, :]))




# ## ADROIT Hand Relocate Problem
# 
# We are focusing on the ADROIT Relocation problem. In this problem we will drain a robotic hand to find a ball, pick it up, and then drop it off at another location. We will train this hand using human generated data provided by Minari. Our agent will be based on the DecisionTransformer approach above - autoregressively outputing the policy needed to solve the environment.
# 
# Ultimately, we want to see how well the resulting Agent can generalize based on training from human data alone.

# ### Action Space
# 
# The ADROIT Hand has 24 Degrees of Freedom. The full action space is documented here: https://robotics.farama.org/envs/adroit_hand/adroit_relocate/
# 
# The action space has 30 dimensions. 24 dimensions for controlling the actuators of the hand, and 6 dimenions for controlling the position of the hand in the environment.
# 
# Inputs to the hands are scaled between to fall in the range [-1, 1]

# ### Observation Space

# The observation space for this system has 39 dimenions. 30 dimensions for describing the state of the arms actuator and the position / rotatation of the arm itself. The final 9 dimensions describe the different in location between the palm of the hand and the ball, the palm of the hand and the target, and finally, the ball and the target. The observation space is documented here: https://robotics.farama.org/envs/adroit_hand/adroit_relocate/

# ### Minari RL Dataset

# We will use the *human* generated data from the D4RL dataset's Relocate Problem. This dataset has 9942 training steps split over 25 training episodes. This data will be used to train the Decision Transformer using an "offline" training approach. We will then evaluate the resulting Decision Transformer on corresponding environment.

# In[ ]:


minari_dataset = minari.load_dataset('D4RL/relocate/human-v2', download=False)


# In[ ]:


print("Observation space:", minari_dataset.observation_space)
print("Action space:", minari_dataset.action_space)
print("Total episodes:", minari_dataset.total_episodes)
print("Total steps:", minari_dataset.total_steps)


# In[ ]:


def process_dataset(dataset=minari_dataset, rtg_scale=1.0):
    """
    Split the observed data into the states, actions, returns_to_go, and timestamps needed for the 
    DecisionTransformer algorithm.
    """
    episodes = list(dataset.iterate_episodes())

    states = []
    actions = []
    returns_to_go = []
    timesteps = []

    # Calculate global min and max for observations to scale to [-1, 1]
    all_obs = np.concatenate([ep.observations for ep in episodes])
    obs_min = np.min(all_obs, axis=0)
    obs_max = np.max(all_obs, axis=0)

    obs_range = np.where(obs_max - obs_min == 0, 1, obs_max - obs_min)

    for ep in episodes:  # Fix: changed self.episodes to episodes
        obs = ep.observations        
        acts = ep.actions 
        rews = ep.rewards

        rtg = np.zeros_like(rews)

        # Calculate the returns to go for each timestep
        for i in reversed(range(len(rews))):
            rtg[i] = rews[i] + (rtg[i+1] if i+1 < len(rews) else 0)

        rtg = rtg / rtg_scale

        states.append(obs)
        actions.append(acts)
        returns_to_go.append(rtg)

        # create the timesteps array for this episode
        timesteps.append(np.arange(len(rews)))

    # Return obs_min and obs_range so they can be used to process inputs during evaluation later
    return states, actions, returns_to_go, timesteps, obs_min, obs_range


# In[ ]:


states, actions, returns_to_go, timestamps, obs_min, obs_range = process_dataset(
    dataset=minari_dataset,
    rtg_scale=3600
)


# In[ ]:


episode_returns = [rtg[0] for rtg in returns_to_go]
average_episode_reward = np.mean(episode_returns)
print(f"Average Reward per Episode: {average_episode_reward:.2f}")


# In[ ]:


class ADROITHandDataset(Dataset):
    def __init__(self, dataset, context_len, samples_per_epoch=400):

        self.context_len = context_len # K from the Paper.
        self.samples_per_epoch = samples_per_epoch
        self.num_human_episodes = 25 

        # get the discretized dataset in the form we need to use with a decision transformer.
        self.states, self.actions, self.returns_to_go, self.timestamps, self.obs_min, self.obs_range = process_dataset(
            dataset=dataset,
            rtg_scale=3600
        )

    def __len__(self):
        # Return the larger number so the DataLoader gives us multiple full batches
        return self.samples_per_epoch

    def __getitem__(self, idx):

        # We use a loop to keep sampling until we find an episode that is long enough to provide a full K-length window.
        # This protects against the agent accidentally generating super short trajectories during online training.
        while True:
            # Sample human experience (first 25 episodes) at least 10% of the time
            if np.random.rand() < 0.10 or len(self.states) <= self.num_human_episodes:
                ep_idx = np.random.randint(0, self.num_human_episodes)
            else:
                # Sample from the agent's online experiences
                ep_idx = np.random.randint(self.num_human_episodes, len(self.states))

            ep_len = len(self.actions[ep_idx])

            # Only accept this episode if it has at least 'context_len' steps
            if ep_len >= self.context_len:
                break

        # Pick a valid starting index that guarantees a full window of 'context_len'
        start_idx = np.random.randint(0, ep_len - self.context_len + 1)
        end_idx = start_idx + self.context_len

        # Extract exactly 'context_len' steps
        states = self.states[ep_idx][start_idx:end_idx]
        actions = self.actions[ep_idx][start_idx:end_idx]
        rtg = self.returns_to_go[ep_idx][start_idx:end_idx]
        timesteps = self.timestamps[ep_idx][start_idx:end_idx]

        # returns (states, actions, return_to_gos, timestamps) as needed by the Decision Transformer.
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rtg, dtype=torch.float32).unsqueeze(1),
            torch.tensor(timesteps, dtype=torch.long)
        )


    def add_trajectory(self, states, actions, rewards):
        """Add a new trajectory to the dataset based on the agents experiences while trying to mimic
        human behaviour."""

        rtg = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            rtg[i] = rewards[i] + (rtg[i+1] if i+1 < len(rewards) else 0)

        self.states.append(states)
        self.actions.append(actions)
        self.returns_to_go.append(rtg)
        self.timestamps.append(np.arange(len(rewards)))

        if len(self.states) > 500:
            # Pop index 25 (the oldest agent-generated trajectory, 
            # since indices 0-24 are the permanent human trajectories)
            self.states.pop(self.num_human_episodes)
            self.actions.pop(self.num_human_episodes)
            self.returns_to_go.pop(self.num_human_episodes)
            self.timestamps.pop(self.num_human_episodes)


# ## Model Configuration

# First we'll need to set some configuration parameters for the DecisionTransformer and its underlying GPT style Transformer:

# In[ ]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# In[ ]:


config = {
   # --- Environment Dimensions ---
    "state_dim": 39,       # Defined by the ADROIT Hand Relocate observation space
    "action_dim": 30,      # Defined by the ADROIT Hand Relocate action space
    "max_ep_length": 1000, # Maximum timesteps in an episode
    # --- Transformer Architecture ---
    "embed_dim": 128,      
    "layers_num": 3,

    # Note: heads_num * head_size must exactly equal embed_dim. 
    "heads_num": 8,        
    "head_size": 16,       # 128 // 4 = 32
    # --- Regularization & Attention ---
    "use_bias": True,
    "dropout_rate": 0.3,   # 0.1 is standard for transformers

    # context_size is the max sequence length passed to the transformer.
    # In Decision Transformers, each timestep uses 3 tokens (Return, State, Action).
    # If your context window (K) of past timesteps is 20, context_size should be 20 * 3 = 60.
    "context_size": 60,

    # simulation parameters
    "target_return": 1.0,

    # training parameters
    "batch_size": 64,
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "epochs": 10000,

    # evaluation parameters
    "evaluation_episodes": 5
}


# In[ ]:


K = config["context_size"] // 3 


# In[ ]:


train_dataset = ADROITHandDataset(minari_dataset, context_len=K)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)


# In[ ]:


model = DecisionTransformer(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])


# In[ ]:


count_parameters(model)


# ## Simulate Behavior

# In[ ]:


import binascii
import binascii
import binascii
import binascii
def simulate_behaviour(model, env, config, obs_min, obs_range, device="cuda"):
    """
    Simulates a single episode with the model and the environment.
    """

    # place model into evaluation mode.
    model.eval()

    # reset the environment
    state, _ = env.reset()

    max_ep_len = config["max_ep_length"]
    # In DT, context_size usually refers to the sequence length of past timesteps
    # If config["context_size"] is 60 tokens (20 timesteps * 3), use 20.
    context_length = config["context_size"] // 3 

    # Initialize history tensors for the transformer's context window
    # we fill them with zeros to start.
    states = torch.zeros((1, max_ep_len, config["state_dim"]), dtype=torch.float32, device=device)
    actions = torch.zeros((1, max_ep_len, config["action_dim"]), dtype=torch.float32, device=device)
    returns_to_go = torch.zeros((1, max_ep_len, 1), dtype=torch.float32, device=device)

    # create the initial timesteps for the simulation
    timesteps = torch.arange(max_ep_len, dtype=torch.long, device=device).unsqueeze(0)

    episode_reward = 0.0


    target_return = config["target_return"]

    for t in range(max_ep_len):
        # discretize the observations
        obs = state

        # Add to state history
        states[0, t] = torch.tensor(obs, dtype=torch.float32, device=device)

        if t == 0:
            returns_to_go[0, t] = torch.tensor([target_return], dtype=torch.float32, device=device)

        # create context; rolls over the history with size context_length
        # context is the Return, state, action triplets for timestamps start_t -> t+1
        start_t = max(0, t - context_length + 1)

        states_input = states[:, start_t:t+1]
        actions_input = actions[:, start_t:t+1]
        returns_input = returns_to_go[:, start_t:t+1]
        timesteps_input = timesteps[:, start_t:t+1]

        # predict the next action
        with torch.no_grad():
            action_preds = model(states_input, actions_input, returns_input, timesteps_input)

            # We want the action prediction for the last timestep in our context
            action = action_preds[0, -1].cpu().numpy()


        # Record the action that the model selected (and environment receives)
        actions[0, t] = torch.tensor(action, dtype=torch.float32, device=device)

        # move the environment forward by one step
        state, reward, terminated, truncated, _ = env.step(action)

        episode_reward += reward

        if terminated or truncated:
                break

        returns_to_go[0, t+1] = torch.tensor([returns_to_go[0, t].item() - reward / 3600], dtype=torch.float32, device=device)

    return episode_reward


# In[ ]:


env  = minari_dataset.recover_environment()


# In[ ]:


test_reward = simulate_behaviour(
    model=model,
    env=env,
    config=config,
    obs_min=obs_min,
    obs_range=obs_range
)


# In[ ]:


print(f"Test Reward: {test_reward}")


# In[ ]:


def evaluate_model(model, env, config, obs_min, obs_range, device="cuda"):
    """
    Evaluates the model over several simulation rounds, responding with the average reward
    """

    total_rewards = 0.0
    for _ in range(config["evaluation_episodes"]):
        r = simulate_behaviour(
            model=model,
            env=env,
            config=config,
            obs_min=obs_min,
            obs_range=obs_range
        )

        total_rewards += r

    return total_rewards / config["evaluation_episodes"]


# ## Visualize Behaviour

# In[ ]:


def visualize_agent(model, dataset=minari_dataset, device="cuda", filename="video/agent_behavior.mp4"):
    """
    Create a visualization of the generative model's policy interacting with the environment.
    """

    render_env = dataset.recover_environment(render_mode="rgb_array")

    # place model into evaluation mode.
    model.eval()

    # reset the environment
    state, _ = render_env.reset()

    max_ep_len = config["max_ep_length"]
    # In DT, context_size usually refers to the sequence length of past timesteps
    # If config["context_size"] is 60 tokens (20 timesteps * 3), use 20.
    context_length = config["context_size"] // 3 

    # Initialize history tensors for the transformer's context window
    # we fill them with zeros to start.
    states = torch.zeros((1, max_ep_len, config["state_dim"]), dtype=torch.float32, device=device)
    actions = torch.zeros((1, max_ep_len, config["action_dim"]), dtype=torch.float32, device=device)
    returns_to_go = torch.zeros((1, max_ep_len, 1), dtype=torch.float32, device=device)

    # create the initial timesteps for the simulation
    timesteps = torch.arange(max_ep_len, dtype=torch.long, device=device).unsqueeze(0)

    episode_reward = 0.0

    discretize_bins = config["discretization_bins"]
    bins = np.linspace(-1.0, 1.0, discretize_bins)

    target_return = config["target_return"]

    frames = []

    for t in range(max_ep_len):
        frames.append(render_env.render())

        obs = state

        # Add to state history
        states[0, t] = torch.tensor(obs, dtype=torch.float32, device=device)

        if t == 0:
            returns_to_go[0, t] = torch.tensor([target_return], dtype=torch.float32, device=device)

        # create context; rolls over the history with size context_length
        # context is the Return, state, action triplets for timestamps start_t -> t+1
        start_t = max(0, t - context_length + 1)

        states_input = states[:, start_t:t+1]
        actions_input = actions[:, start_t:t+1]
        returns_input = returns_to_go[:, start_t:t+1]
        timesteps_input = timesteps[:, start_t:t+1]

        # predict the next action
        with torch.no_grad():
            action_preds = model(states_input, actions_input, returns_input, timesteps_input)

            # We want the action prediction for the last timestep in our context
            action = action_preds[0, -1].cpu().numpy()

        # Record the action that the model selected (and environment receives)
        actions[0, t] = torch.tensor(action, dtype=torch.float32, device=device)

        # move the environment forward by one step
        state, reward, terminated, truncated, _ = render_env.step(action)

        episode_reward += reward

        if terminated or truncated:
            break

        returns_to_go[0, t+1] = torch.tensor([returns_to_go[0, t].item() - reward / 3600], dtype=torch.float32, device=device)


    render_env.close()

    # Save the captured frames as a video
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved video to {filename}")

    # Display it in the notebook
    return Video(filename, embed=True)


# In[ ]:


visualize_agent(model=model, filename="video/pre_training_behaviour.mp4")


# ## Training Loop

# In[ ]:


def train_one_epoch(model, optimizer, train_loader, device="cuda", scheduler=None):
    """
    Trains the model for a single epoch
    """

    model.train()
    total_loss = 0.0

    for states, actions, rtg, timesteps in train_loader:
        states = states.to(device)
        actions = actions.to(device)
        rtg = rtg.to(device)
        timesteps = timesteps.to(device)

        optimizer.zero_grad()
        action_preds = model(states, actions, rtg, timesteps)

        loss = F.mse_loss(action_preds, actions)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    average_epoch_loss = total_loss / len(train_loader)

    return average_epoch_loss


# In[ ]:


train_one_epoch(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader
)


# In[ ]:


def train_model(model, config, optimizer, train_loader, device="cuda", scheduler=None):
    """
    Trains the model for the given number of epoches and returns relevlant statistics.
    """
    losses = []
    rewards = []

    best_reward = -float('inf')

    for i in range(config["epochs"]):
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            scheduler=scheduler
        )

        losses.append(loss)

        if i % 10 == 0:

            r = evaluate_model(
                model=model,
                env=env,
                config=config,
                obs_min=obs_min,
                obs_range=obs_range
            )

            rewards.append(r)

            if r > best_reward:
                best_reward = r
                torch.save(model.state_dict(), "best_decision_transformer.pth")

            print(f"\rEpoch {i} Loss {loss} Current Reward: {r}, Best Reward: {best_reward}", end="")

    return losses, rewards


# In[ ]:


losses = train_model(
    model=model,
    config=config,
    optimizer=optimizer,
    train_loader=train_loader
)


# In[ ]:


# Load the best model weights saved by early stopping
try:
    model.load_state_dict(torch.load("best_decision_transformer.pth", weights_only=True))
    print("Loaded best model weights!")
except FileNotFoundError:
    print("No best model found, using current weights.")


# In[ ]:


test_reward = simulate_behaviour(
    model=model,
    env=env,
    config=config,
    obs_min=obs_min,
    obs_range=obs_range
)


# In[ ]:


print(f"Test Reward: {test_reward}")


# In[ ]:


visualize_agent(model=model)


# ## Results

# 
