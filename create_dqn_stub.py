#!/usr/bin/env python
"""Quick workaround: Create minimal DQN weights just for inference."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Build the same model as in train_dqn.py
model = Sequential()
model.add(Flatten(input_shape=(1, 10)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='linear'))
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

# Do a single forward pass to initialize weights
dummy_data = np.random.randn(1, 1, 10).astype(np.float32)
_ = model.predict(dummy_data, verbose=0)

# Save weights
os.makedirs('models', exist_ok=True)
model.save_weights('models/dqn_weights.weights.h5')
print('✓ Minimal DQN weights created at models/dqn_weights.weights.h5')
print('NOTE: These are untrained weights. For better results, run:')
print('      python train_dqn.py --nb-steps 500')
