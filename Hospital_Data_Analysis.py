import numpy as np 

# Generate data
bmi = np.random.randint(18, 40, 200)
age = np.random.randint(20, 80, 200)
glucose = np.random.randint(70, 200, 200)

# Create risk labels correctly (convert to scalars for comparison)
risk = np.array([1 if (bmi[x].item() > 30 and glucose[x].item() > 140) or 
                 (age[x].item() > 50 and glucose[x].item() > 130) 
                 else 0 for x in range(200)])

# Reshape and combine features
bmi = bmi.reshape(-1, 1)
age = age.reshape(-1, 1)
glucose = glucose.reshape(-1, 1)
data = np.concatenate([bmi, age, glucose], axis=1)

# Split into train and test
data1 = data[0:160]  # 160x3
data2 = data[160:200]  # 40x3
risk1 = risk[0:160].reshape(-1, 1)  # 160x1
risk2 = risk[160:200].reshape(-1, 1)  # 40x1

# Add bias column (column of ones) to the LEFT of features
# Training data
bias_col = np.ones((160, 1))  # Column of ones for bias
X_train = np.concatenate([bias_col, data1], axis=1)  # Now 160x4 [bias, bmi, age, glucose]

# Test data
bias_col_test = np.ones((40, 1))
X_test = np.concatenate([bias_col_test, data2], axis=1)  # 40x4

# 1. Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Initialize weights (for 4 columns: bias + 3 features)
weights = np.zeros((4, 1))  # [w0 (bias), w1 (bmi), w2 (age), w3 (glucose)]

# 3. Gradient descent training
learning_rate = 0.1
iterations = 1000

for i in range(iterations):
    # Forward pass
    z = np.dot(X_train, weights)
    predictions = sigmoid(z)
    
    # Calculate loss (cross-entropy with epsilon to avoid log(0))
    epsilon = 1e-15
    predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.mean(risk1 * np.log(predictions_clipped) + 
                    (1 - risk1) * np.log(1 - predictions_clipped))
    
    # Calculate gradient
    gradient = np.dot(X_train.T, (predictions - risk1)) / len(risk1)
    
    # Update weights
    weights -= learning_rate * gradient
    
    # Print progress
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss:.4f}")

# 5. Make predictions function
def predict(X):
    z = np.dot(X, weights)
    probabilities = sigmoid(z)
    return (probabilities >= 0.5).astype(int)

# Test on training data
train_predictions = predict(X_train)
train_accuracy = np.mean(train_predictions == risk1) * 100
print(f"\nTraining Accuracy: {train_accuracy:.2f}%")

# Test on test data
test_predictions = predict(X_test)
test_accuracy = np.mean(test_predictions == risk2) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Print weights
print("\nLearned weights:")
print(f"Bias (w0): {weights[0][0]:.4f}")
print(f"BMI (w1): {weights[1][0]:.4f}")
print(f"Age (w2): {weights[2][0]:.4f}")
print(f"Glucose (w3): {weights[3][0]:.4f}")

# Example prediction
print("\nExample predictions:")
for i in range(5):
    print(f"Patient {i+1}: BMI={data2[i][0]}, Age={data2[i][1]}, Glucose={data2[i][2]}")
    print(f"  Actual risk: {risk2[i][0]}, Predicted risk: {test_predictions[i][0]}")