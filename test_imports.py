#!/usr/bin/env python
import sys

print("Testing imports...\n")

try:
    import gym
    print("✓ gym")
except Exception as e:
    print(f"✗ gym: {e}")

try:
    import tensorflow as tf
    print(f"✓ tensorflow ({tf.__version__})")
except Exception as e:
    print(f"✗ tensorflow: {e}")

try:
    from tensorflow.keras.models import Sequential
    print("✓ tensorflow.keras.models.Sequential")
except Exception as e:
    print(f"✗ tensorflow.keras.models.Sequential: {e}")

try:
    from tensorflow.keras.layers import Dense, Flatten
    print("✓ tensorflow.keras.layers (Dense, Flatten)")
except Exception as e:
    print(f"✗ tensorflow.keras.layers: {e}")

try:
    from rl.agents.dqn import DQNAgent
    print("✓ rl.agents.dqn.DQNAgent")
except Exception as e:
    print(f"✗ rl.agents.dqn.DQNAgent: {e}")

try:
    from rl.policy import EpsGreedyQPolicy
    print("✓ rl.policy.EpsGreedyQPolicy")
except Exception as e:
    print(f"✗ rl.policy.EpsGreedyQPolicy: {e}")

try:
    from rl.memory import SequentialMemory
    print("✓ rl.memory.SequentialMemory")
except Exception as e:
    print(f"✗ rl.memory.SequentialMemory: {e}")

print("\nAll imports checked.")
