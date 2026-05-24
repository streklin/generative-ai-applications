# Generative AI Applications: Decision Transformers for Robotic Manipulation

**Author:** Gene Foxwell  
**Course:** Udacity AI Masters Project #5  

## Overview
This project explores how Generative Pretrained Transformer (GPT) architectures can be adapted to solve complex Reinforcement Learning (RL) problems. Based on the [Decision Transformer](https://arxiv.org/abs/2106.01345) framework, this project frames RL as a sequence modeling problem. 

Specifically, we train an agent to solve the **Adroit Hand Relocate** task—controlling a highly complex 24 Degrees of Freedom (DoF) robotic hand to locate a ball, pick it up, and transport it to a specific target location. 

The agent is trained entirely offline using human demonstration data from the [Minari D4RL Dataset](https://minari.farama.org/environments/d4rl/adroit_relocate/) (`D4RL/relocate/human-v2`).

## Key Innovations & Techniques

Training a sequence model purely on offline data often leads to severe overfitting and poor out-of-distribution (OOD) generalization. To combat this while strictly adhering to academic requirements (no synthetic/AI-generated datasets), this project implements two major algorithmic improvements:

1. **Semantic Hindsight Experience Replay (HER):** 
   Instead of conditioning the model exclusively on the global return of full episodes, the data loader dynamically isolates sub-trajectories on the fly. By calculating physical distances directly from the 39-dimensional observation space, the system identifies exact semantic milestones (e.g., reaching the ball, bringing the ball to the target). We then slice the trajectory and relabel the *Return-to-Go* (RTG) specifically for that sub-task, teaching the model robust intermediate behaviors.
   
2. **Auxiliary Reward Prediction:** 
   Standard Decision Transformers rely purely on Behavior Cloning (MSE loss on the action space). In this project, an auxiliary prediction head is added to the transformer to explicitly predict the immediate reward $r_t$. This acts as a powerful regularizer, forcing the internal attention mechanism to capture features that strongly correlate with environmental rewards.

## Architecture
* **Backbone:** GPT-style Transformer
* **Sequence Input:** Triplets of `(Return-to-Go, State, Action)`
* **Context Window:** 30 tokens (10 timesteps)
* **Embedding Dimension:** 64 (3 layers, 4 attention heads)
* **Action Space:** 30 Continuous Dimensions (Scaled to `[-1, 1]`)
* **Observation Space:** 39 Dimensions (Includes 9 dimensions of spatial relational coordinates)

## Getting Started

### Prerequisites
Install the required packages strictly aligned with the notebook execution environment:

```bash
pip install -r requirements.txt
```

### Running the Project
The entire pipeline—from data ingestion and processing to model training and visualization—is contained within the primary Jupyter Notebook.

1. Open `generative_model.ipynb`.
2. Run the cells sequentially to initialize the Transformer model, download the Minari dataset, and execute the training loop.
3. The notebook includes a `visualize_agent()` function which renders the environment and saves `.mp4` videos of the agent's performance in the `video/` directory.

## References
* **Decision Transformer:** [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) (Chen et al., 2021)
* **Environment:** [Adroit Dexterous Manipulation](https://robotics.farama.org/envs/adroit_hand/adroit_relocate/)
* **Dataset:** [Minari D4RL Human Datasets](https://minari.farama.org/environments/d4rl/adroit_relocate/)
