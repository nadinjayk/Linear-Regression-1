import pandas as pd
import matplotlib.pyplot as plt
import math

# Load CSV
data = pd.read_csv("Housing.csv")

# Preprocessing
data = data[["area", "price"]]
data["area"] = pd.to_numeric(data["area"], errors="coerce")
data["price"] = pd.to_numeric(data["price"], errors="coerce")
data = data.dropna()

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Train-test split: 80% train, 20% test
split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].area
        y = points.iloc[i].price
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].area
        y = points.iloc[i].price

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b


def mean_absolute_error(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].area
        y = points.iloc[i].price
        total_error += abs(y - (m * x + b))
    return total_error / float(len(points))


def root_mean_squared_error(m, b, points):
    return math.sqrt(loss_function(m, b, points))


def r2_score(m, b, points):
    y_mean = points["price"].mean()

    ss_total = 0
    ss_residual = 0

    for i in range(len(points)):
        x = points.iloc[i].area
        y = points.iloc[i].price
        y_pred = m * x + b

        ss_total += (y - y_mean) ** 2
        ss_residual += (y - y_pred) ** 2

    return 1 - (ss_residual / ss_total)


m = 0
b = 0
L = 0.00000001
epochs = 1000

prev_loss = float("inf")

for i in range(epochs):
    m, b = gradient_descent(m, b, train_data, L)
    train_loss = loss_function(m, b, train_data)

    if (i + 1) % 100 == 0 or i == 0:
        print(f"epoch {i+1}/{epochs} | train loss = {train_loss:.6f} | m = {m:.6f} | b = {b:.6f}")

    if abs(prev_loss - train_loss) < 1e-3:
        print("Stopping early: training loss is changing very little now.")
        break

    prev_loss = train_loss


# Final metrics
train_mse = loss_function(m, b, train_data)
test_mse = loss_function(m, b, test_data)

train_rmse = root_mean_squared_error(m, b, train_data)
test_rmse = root_mean_squared_error(m, b, test_data)

train_mae = mean_absolute_error(m, b, train_data)
test_mae = mean_absolute_error(m, b, test_data)

train_r2 = r2_score(m, b, train_data)
test_r2 = r2_score(m, b, test_data)

print("\nFinal parameters:")
print("m =", m)
print("b =", b)

print("\nDataset sizes:")
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

print("\nTraining metrics:")
print(f"MSE  = {train_mse:.6f}")
print(f"RMSE = {train_rmse:.6f}")
print(f"MAE  = {train_mae:.6f}")
print(f"R2   = {train_r2:.6f}")

print("\nTest metrics:")
print(f"MSE  = {test_mse:.6f}")
print(f"RMSE = {test_rmse:.6f}")
print(f"MAE  = {test_mae:.6f}")
print(f"R2   = {test_r2:.6f}")

# Predict price for a given area
given_area = 5000
predicted_price = m * given_area + b
print(f"\nPredicted price for area {given_area} = {predicted_price:.2f}")

# Plot training and test data
plt.scatter(train_data.area, train_data.price, color="blue", alpha=0.4, label="Train data")
plt.scatter(test_data.area, test_data.price, color="green", alpha=0.4, label="Test data")

# Plot regression line using full data range
x_min = data.area.min()
x_max = data.area.max()
x_vals = [x for x in range(int(x_min), int(x_max) + 1, 100)]

plt.plot(x_vals, [m * x + b for x in x_vals], color="red", label="Regression line")

plt.xlabel("area")
plt.ylabel("price")
plt.legend()
plt.show()
