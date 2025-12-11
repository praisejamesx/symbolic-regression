# Symbolic Regression with Genetic Programming

A symbolic regression system that discovers mathematical expressions from data using evolutionary algorithms.

## Overview

This system implements genetic programming to automatically discover mathematical expressions that fit given data. It evolves expression trees through natural selection, optimizing both accuracy and simplicity.

**Key Results Example:** When fitting data from the function `f(x) = sin(x)*x + 0.5*exp(-0.1*x²)`, the system discovered `(x * sin(x)) + 0.2708` with **R² = 0.9955** in 59 generations.

## Key Features

- **Vectorized Evaluation**: NumPy-optimized expression evaluation for high performance
- **Automatic Constant Optimization**: Gradient-based refinement of numeric constants using L-BFGS-B
- **Intelligent Evolution**: Tournament selection, elitism, and diversity maintenance
- **Bloat Control**: Parsimony pressure to prevent overly complex expressions
- **Early Stopping**: Automatic termination when performance plateaus
- **Live Visualization**: Real-time plots of evolution progress (optional)
- **Model Persistence**: Save/load discovered expressions
- **Robust Handling**: Graceful management of invalid operations (division by zero, log(negative), etc.)

## Installation

```bash
# Clone the repository
git clone https://github.com/praisejamesx/symbolic-regression.git
cd symbolic-regression

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.9.0
sympy>=1.11.0
tqdm>=4.64.0
```

## Quick Start

```python
import numpy as np
from symbolic_regression import SymbolicRegressor

# Generate some data
x = np.linspace(-5, 5, 100)
y_true = np.sin(x) * x + 0.5 * np.exp(-0.1 * x**2)
y_data = y_true + np.random.normal(0, 0.1, len(x))  # Add noise

# Create regressor
sr = SymbolicRegressor(
    operators=['+', '-', '*', '/', 'sin', 'exp', 'log', 'cos'],
    population_size=200,
    max_depth=6,
    parsimony_coefficient=0.01
)

# Run evolution
best_expression, history = sr.fit(x, y_data, y_true, generations=100)
print(f"Discovered: {best_expression}")

# Use for predictions
predictions = sr.predict(x)
func = sr.to_function()
print(f"f(π) ≈ {func(3.14159)}")
```

## API Reference

### SymbolicRegressor Class

```python
SymbolicRegressor(
    operators: List[str] = None,           # Mathematical operators to use
    max_depth: int = 6,                    # Maximum tree depth
    population_size: int = 100,            # Size of population
    parsimony_coefficient: float = 0.01,   # Complexity penalty coefficient
    crossover_rate: float = 0.7,           # Probability of crossover
    mutation_rate: float = 0.3,            # Probability of mutation
    tournament_size: int = 3,              # Size for tournament selection
    elite_size: int = 5,                   # Number of elite to preserve
    early_stopping_patience: int = 20,     # Stop after N gens without improvement
    random_state: int = None               # Random seed for reproducibility
)
```

### Available Operators
- Binary: `+`, `-`, `*`, `/`, `^` (power)
- Unary: `sin`, `cos`, `exp`, `log`, `sqrt`

## Algorithm Details

### How It Works

1. **Initialization**: Creates random expression trees using "grow" and "full" methods
2. **Evaluation**: Computes fitness = -(MSE + λ × complexity) for each individual
3. **Selection**: Tournament selection chooses parents based on fitness
4. **Crossover**: Exchanges random subtrees between parents
5. **Mutation**: Replaces random subtrees with newly generated ones
6. **Constant Optimization**: Gradient-based refinement of numeric constants
7. **Elitism**: Preserves top performers unchanged
8. **Termination**: Stops after specified generations or early stopping

### Fitness Function

```
Fitness = -[MSE + λ × (size + 0.3 × depth)]
```
Where λ is the parsimony coefficient (controls complexity penalty).

## Advanced Usage

### Custom Problem

```python
# Define your target function
def target_function(x):
    return 2.5 * np.sin(1.7 * x) + 0.8 * np.cos(3.2 * x)

x = np.linspace(-3, 3, 200)
y_true = target_function(x)
y_data = y_true + np.random.normal(0, 0.05, len(x))

# Configure for your problem
sr = SymbolicRegressor(
    operators=['+', '-', '*', '/', 'sin', 'cos'],
    max_depth=7,
    population_size=300,
    parsimony_coefficient=0.001,  # Allow more complexity
    early_stopping_patience=30    # Be more patient
)

# Run with live visualization
best_expr, history = sr.fit(
    x, y_data, y_true,
    generations=150,
    show_progress=True,
    live_plot=True  # Watch evolution in real-time
)
```

### Model Persistence

```python
# Save discovered model
sr.save_model("discovered_model.pkl")

# Load and use later
sr_loaded = SymbolicRegressor.load_model("discovered_model.pkl")
predictions = sr_loaded.predict(new_x)

# Convert to callable function
func = sr_loaded.to_function()
```

## Performance Characteristics

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| Population Size | 100-500 | Larger = more diversity, slower |
| Generations | 50-200 | More = better results, slower |
| Parsimony λ | 0.001-0.1 | Higher = simpler expressions |
| Early Stopping | 10-30 | Higher = more exploration |

**Typical Performance**: For 100 data points, population of 200, and 100 generations:
- Runtime: 2-10 seconds
- Memory: 50-200 MB
- Accuracy: R² > 0.99 for well-defined problems

## Practical Applications

### Scientific Discovery
```python
# Discover physical laws from experimental data
sr = SymbolicRegressor(operators=['+', '-', '*', '/', 'sqrt'])
# Input: pendulum lengths, Output: periods
# System might discover: T ≈ 2π√(L/g)
```

### Feature Engineering
```python
# Generate interpretable features for ML models
sr = SymbolicRegression(operators=['+', '-', '*', '/', 'log', 'exp'])
# Use discovered expressions as features in your ML pipeline
```

### Curve Fitting
```python
# Fit noisy experimental data with interpretable model
sr = SymbolicRegressor(operators=['+', '-', '*', '/', 'sin', 'cos', 'exp'])
# Get human-readable formula instead of black-box spline
```

## Comparison with Other Methods

| Method | Interpretability | Performance | Flexibility |
|--------|-----------------|-------------|-------------|
| **Symbolic Regression** | Excellent | Good | High |
| Neural Networks | Poor | Excellent | High |
| Polynomial Regression | Good | Good | Low |
| Splines | Moderate | Good | High |

## Troubleshooting

### Common Issues & Solutions

1. **No Convergence**
   ```python
   # Increase population size and generations
   sr = SymbolicRegressor(population_size=500, generations=200)
   ```

2. **Overly Complex Expressions**
   ```python
   # Increase parsimony coefficient
   sr = SymbolicRegressor(parsimony_coefficient=0.1)
   ```

3. **Missing Operators**
   ```python
   # Add needed operators
   sr = SymbolicRegressor(operators=['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log'])
   ```

4. **Slow Performance**
   ```python
   # Reduce population size or use fewer operators
   sr = SymbolicRegressor(population_size=100, operators=['+', '-', '*', '/'])
   ```

## Recommended Reading

For those interested in diving deeper into symbolic regression, I highly recommend:

**"Symbolic Regression"** by Gabriel Kronberger, Bogdan Burlacu, Michael Kommenda, Stephan M. Winkler, and Michael Affenzeller.

This comprehensive book covers:
- Foundations of genetic programming
- Advanced symbolic regression techniques
- Real-world applications and case studies
- State-of-the-art algorithms and implementations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use this code in research, please cite:

```bibtex
@software{symbolic_regression_gp,
  title = {Symbolic Regression with Genetic Programming},
  author = {Praise James},
  year = {2025},
  url = {https://github.com/praisejamesx/symbolic-regression}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by Koza's work on genetic programming
- Uses efficient NumPy vectorization for performance
- Gradient-based constant optimization adapted from scipy
- Visualization with matplotlib

---

**Note**: This system is designed for both research and practical applications. For production use, consider adding input validation, logging, and monitoring for long-running evolutions.

For questions or support, please open an issue on GitHub.
