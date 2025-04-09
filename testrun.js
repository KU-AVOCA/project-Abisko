import numpy as np

# Set random seed for reproducibility
np.random.seed(42)  # Optional - remove for different splits each time

# Create array of IDs from 1 to 40
ids = np.arange(1, 41)

# Shuffle the IDs randomly
np.random.shuffle(ids)

# Calculate split point (2/3 for training)
split_idx = int(len(ids) * 2/3)

# Split into training and testing sets
train_ids = ids[:split_idx]
test_ids = ids[split_idx:]

print(f"Training set size: {len(train_ids)}")
print(f"Testing set size: {len(test_ids)}")
print(f"Training IDs: {sorted(train_ids)}")
print(f"Testing IDs: {sorted(test_ids)}")