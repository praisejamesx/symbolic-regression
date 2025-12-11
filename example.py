"""
Example usage of SymbolicRegressor
"""

import numpy as np
from symbolic_regression import SymbolicRegressor, create_example_data

def main():
    # Create example data
    x, y_data, y_true = create_example_data()
    
    # Create regressor
    sr = SymbolicRegressor(
        operators=['+', '-', '*', '/', 'sin', 'exp', 'log', 'cos'],
        max_depth=6,
        population_size=200,
        parsimony_coefficient=0.01,
        early_stopping_patience=20,
        random_state=42
    )
    
    # Run evolution
    best_expression, fitness_history = sr.fit(
        x, y_data, y_true, 
        generations=100,
        show_progress=True,
        live_plot=False
    )
    
    # Make predictions
    predictions = sr.predict(x)
    print(f"\nPredictions shape: {predictions.shape}")
    
    # Get as function
    func = sr.to_function()
    print(f"f(3.14) = {func(3.14):.4f}")
    
    return sr

if __name__ == "__main__":
    sr = main()