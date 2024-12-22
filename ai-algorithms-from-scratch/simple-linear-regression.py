


import numpy as np # type: ignore

# Generate or load data
X = np.array([1, 2, 3, 4, 5])  # Example input features
y = np.array([2.2, 2.8, 4.5, 3.9, 5.1])  # Example target variable

# Initialize parameters
w = 0.0  # initial weight
b = 0.0  # initial bias
learning_rate = 0.01
num_iterations = 1000

# Function to calculate mean squared error
def compute_cost(X, y, w, b):
    m = len(X)
    total_cost = np.sum((X * w + b - y) ** 2) / (2 * m)
    return total_cost

# Gradient descent function
def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    m = len(X)
    for i in range(num_iterations):
        # Calculate predictions
        y_pred = X * w + b
        
        # Calculate gradients
        dw = np.sum((y_pred - y) * X) / m
        db = np.sum(y_pred - y) / m
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Print cost every 100 iterations for tracking
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iteration {i}: Cost {cost:.4f}")
    
    return w, b

# Train the model
w, b = gradient_descent(X, y, w, b, learning_rate, num_iterations)

# Print the learned parameters
print(f"Trained weight: {w:.4f}")
print(f"Trained bias: {b:.4f}")

# Function to make predictions
def predict(X, w, b):
    return X * w + b

# Make predictions on new data
new_X = np.array([6, 7, 8])
predictions = predict(new_X, w, b)
print("Predictions:", predictions)
