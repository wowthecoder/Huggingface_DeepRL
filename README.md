# 🤗 Hugging Face Deep Reinforcement Learning course
This repository contains my code solutions to the hands-on projects in the [Hugging Face DeepRL course](https://huggingface.co/learn/deep-rl-course/unit0/introduction).
All the models are published on my [Hugging Face profile](https://huggingface.co/wowthecoder) as well.

## Unit 1: Introduction to Deep Reinforcement Learning
- **Overview**: Covers the basics of Reinforcement Learning (RL) and introduces the concept of Deep RL.
- **Key Concepts**:
  - RL frameworks and terminology.
  - Exploration vs. exploitation trade-off.
  - Introduction to the OpenAI Gym.
- **Hands-on**: Training a PPO agent on the LunarLander-v2 environment using Stable-Baselines3 and uploading it to the Hugging Face Hub.

## Bonus Unit 1: Introduction to Deep Reinforcement Learning with Huggy
- **Hands-on**: Training Huggy the dog to fetch a stick in a Unity ML Agents environment.

## Unit 2: Introduction to Q-Learning
- **Overview**: Delves into value-based methods, focusing on the Q-Learning algorithm.
- **Key Concepts**:
  - Difference between Monte Carlo and Temporal Difference Learning.
  - Bellman equation for state/state-action value calculation
- **Hands-on**:
  - Implementing the Q-Learning algorithm from scratch
  - Training the agents in FrozenLake-v1 and Taxi environments.

## Unit 3: Deep Q-Learning with Atari Games
- **Overview**: Transition from Q-Learning to Deep Q-Networks (DQN) for handling high-dimensional input spaces.
- **Key Concepts**:
  - Understanding the architecture of DQNs.
  - Experience replay and target networks.
- **Hands-on**: Training a DQN agent to play Atari games like Space Invaders using RL-Baselines3 Zoo.

## Bonus Unit 2: Automatic Hyperparameter Tuning with Optuna
- **Overview**: Introduces Optuna library for automating hyperparameter optimization in Deep RL models for better performance.
- **Hands-on**:
  - Tune hyperparameters of a Soft Actor Critic (SAC) model on the Pendulum environment.
  - Tune hyperparameters of a Deep Q-Network model on the Space Invaders environment. 

## Unit 4: Policy Gradient with PyTorch
- **Overview**: Explores policy-based methods, focusing on the Monte Carlo REINFORCE algorithm.
- **Key Concepts**:
  - Understanding policy gradients method and its pros and cons.
  - Deep dive into Monte Carlo REINFORCE algorithm and the equations involved. 
- **Hands-on**:
  - Implementing the REINFORCE algorithm from scratch, and training an agent in the CartPole-v1 environment.
  - Attempted the same thing on the Pixelcopter environment but encountered the large reward variance issue 
  - Implemented Advantage Actor Critic algorithm to lower the variance by using a value function (critic) to estimate a baseline

## Unit 5: Introduction to Unity ML-Agents
- **Overview**: Introduces Unity ML-Agents for creating complex and customizable environments.
- **Hands-on**: 
  - Training an agent to move and shoot a target repeatedly
  - Training an agent to navigate a maze and knock down pyramids

## Unit 6: Actor-Critic Methods with Robotics Environments
- **Overview**: Combines value-based and policy-based methods through Actor-Critic algorithms.
- **Key Concepts**:
  - Reason behind problem of variance in vanilla REINFORCE
  - Understanding the Actor-Critic architecture
- **Hands-on**: Training a robotic arm to pick up objects in the PandaReachDense-v3 simulation using Panda Gym.

## Unit 7: Introduction to Multi-Agents and AI vs AI
- **Overview**: Explores multi-agent systems in competitive/cooperative scenarios.
- **Key Concepts**: Understanding the centralized and decentralized training approaches, and the pros and cons of each.
- **Hands-on**: Training AI agents to play 2v2 soccer(where the agent has to both cooperate with its teammate and also compete with the 2 opponents) in a Unity ML Agents environment. 

## Unit 8 Part 1: Proximal Policy Optimization (PPO)
- **Overview**: Introduces PPO, a stable and efficient policy optimization algorithm.
- **Key Concepts**:
  - Intuition behind PPO, which is to avoid policy updates that are too large to stabilise training.
  - Clipped objective function.
- **Hands-on**:
  - Write the PPO algorithm from scratch with reference to the CleanRL library
  - Train the agent on the LunarLander-v2 environment and log the metrics using Weights and Biases integration

## Unit 8 Part 2: Proximal Policy Optimization (PPO) with Doom
- **Overview**: Introduces Sample Factory, an asynchronous implementation of the PPO algorithm, train our agent to play vizdoom (an open source version of Doom).
- **Hands-on**: Training agents to navigate and perform tasks in Doom scenarios, such as the Health Gathering level, where the agent must collect health packs to avoid dying.

# 📌 Notes
This repository is a personal endeavor to reinforce my understanding of Deep RL concepts.

The implementations are based on the Hugging Face course materials and may include additional experiments or modifications.
