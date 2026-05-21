# cmeuncerpy

A Python package for machine learning models with PyTorch and Rust backends. Features high-performance implementations of Linear Regression and Neural Network models.

## Installation

```bash
pip install cmeuncerpy
```

## Features

- **Linear Regression**: PyTorch-based implementation with gradient descent optimization
- **Neural Network Regression**: Flexible multi-layer neural network for regression tasks
- **Rust Integration**: Performance-critical components implemented in Rust via PyO3

## Quick Start

```python
from cmeuncerpy.models import LinearRegression
import numpy as np

# Create model
model = LinearRegression(no_features=2, learning_rate=0.01, max_epochs=100)

# Generate sample data
X_train = np.random.randn(100, 2)
y_train = np.random.randn(100)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_train)
```

## Requirements

- Python 3.9 or higher
- PyTorch 2.0.0+
- NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

## Author

Syed Raza (alizeejah972@gmail.com)

## License

MIT