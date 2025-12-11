"""
Symbolic Regression with Genetic Programming
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import random
import math
import pickle
from scipy.optimize import minimize
import sympy as sp
from tqdm import tqdm
import warnings
import sys
import os

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    import multiprocessing
    multiprocessing.freeze_support()

class NodeType(Enum):
    """Types of nodes in expression tree."""
    OPERATOR = 1
    VARIABLE = 2
    CONSTANT = 3

@dataclass
class Node:
    """Expression tree node with vectorized evaluation."""
    node_type: NodeType
    value: Any = None
    children: List['Node'] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.node_type == NodeType.CONSTANT and isinstance(self.value, str):
            try:
                self.value = float(self.value)
            except (ValueError, TypeError):
                self.value = 0.0
    
    def evaluate_vectorized(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate expression for vector of x values.
        Uses NumPy vectorization for performance.
        """
        try:
            if self.node_type == NodeType.VARIABLE:
                return x
            elif self.node_type == NodeType.CONSTANT:
                return np.full_like(x, self.value, dtype=np.float64)
            else:
                # Operator node
                if self.value == '+':
                    return self.children[0].evaluate_vectorized(x) + self.children[1].evaluate_vectorized(x)
                elif self.value == '-':
                    return self.children[0].evaluate_vectorized(x) - self.children[1].evaluate_vectorized(x)
                elif self.value == '*':
                    return self.children[0].evaluate_vectorized(x) * self.children[1].evaluate_vectorized(x)
                elif self.value == '/':
                    denom = self.children[1].evaluate_vectorized(x)
                    # Safe division with clipping
                    denom = np.where(np.abs(denom) < 1e-10, np.sign(denom) * 1e-10, denom)
                    return self.children[0].evaluate_vectorized(x) / denom
                elif self.value == 'sin':
                    return np.sin(self.children[0].evaluate_vectorized(x))
                elif self.value == 'cos':
                    return np.cos(self.children[0].evaluate_vectorized(x))
                elif self.value == 'exp':
                    val = self.children[0].evaluate_vectorized(x)
                    return np.exp(np.clip(val, -100, 100))
                elif self.value == 'log':
                    val = self.children[0].evaluate_vectorized(x)
                    val = np.where(val <= 0, 1e-10, val)
                    return np.log(val)
                elif self.value == 'sqrt':
                    val = self.children[0].evaluate_vectorized(x)
                    val = np.where(val < 0, 0, val)
                    return np.sqrt(val)
                elif self.value == '^':
                    base = self.children[0].evaluate_vectorized(x)
                    exp = self.children[1].evaluate_vectorized(x)
                    return np.power(np.abs(base), exp) * np.sign(base) ** exp
                else:
                    raise ValueError(f"Unknown operator: {self.value}")
        except Exception:
            return np.full_like(x, np.nan)
    
    @property
    def size(self) -> int:
        """Number of nodes in tree."""
        if not self.children:
            return 1
        return 1 + sum(child.size for child in self.children)
    
    @property
    def depth(self) -> int:
        """Depth of tree."""
        if not self.children:
            return 1
        return 1 + max(child.depth for child in self.children)
    
    def collect_constants(self) -> List['Node']:
        """Collect all constant nodes in tree."""
        constants = []
        if self.node_type == NodeType.CONSTANT:
            constants.append(self)
        for child in self.children:
            constants.extend(child.collect_constants())
        return constants
    
    def copy(self) -> 'Node':
        """Create deep copy of tree."""
        return Node(
            node_type=self.node_type,
            value=self.value,
            children=[child.copy() for child in self.children]
        )
    
    def __str__(self) -> str:
        """String representation."""
        if self.node_type == NodeType.VARIABLE:
            return "x"
        elif self.node_type == NodeType.CONSTANT:
            return f"{self.value:.4f}"
        else:
            if self.value in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                return f"{self.value}({self.children[0]})"
            elif self.value == '^':
                return f"({self.children[0]})^{self.children[1]}"
            else:
                return f"({self.children[0]} {self.value} {self.children[1]})"

class SymbolicRegressor:
    """Main symbolic regression class."""
    
    OPERATOR_ARITY = {
        '+': 2, '-': 2, '*': 2, '/': 2,
        'sin': 1, 'cos': 1, 'exp': 1, 'log': 1, 'sqrt': 1,
        '^': 2
    }
    
    def __init__(
        self,
        operators: List[str] = None,
        max_depth: int = 6,
        population_size: int = 100,
        parsimony_coefficient: float = 0.01,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        elite_size: int = 5,
        early_stopping_patience: int = 20,
        random_state: int = None
    ):
        """
        Initialize symbolic regressor.
        
        Args:
            operators: List of operator symbols to use
            max_depth: Maximum tree depth
            population_size: Size of population
            parsimony_coefficient: Complexity penalty coefficient
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament selection
            elite_size: Number of elite individuals preserved
            early_stopping_patience: Stop if no improvement for N generations
            random_state: Random seed for reproducibility
        """
        self.operators = operators or ['+', '-', '*', '/', 'sin', 'exp']
        self.max_depth = max_depth
        self.population_size = population_size
        self.parsimony_coefficient = parsimony_coefficient
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.early_stopping_patience = early_stopping_patience
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
        
        # State
        self.population = []
        self.fitness_history = []
        self.best_individuals = []
        self.complexity_history = []
        self.generation = 0
        
        # Cache for evaluation
        self._eval_cache = {}
    
    def _generate_random_tree(self, depth: int = 0, method: str = 'grow') -> Node:
        """Generate random expression tree."""
        if depth >= self.max_depth or (method == 'grow' and random.random() < 0.3):
            # Terminal node
            if random.random() < 0.5:
                return Node(NodeType.VARIABLE, 'x')
            else:
                return Node(NodeType.CONSTANT, random.uniform(-5, 5))
        else:
            # Internal node
            op = random.choice(self.operators)
            arity = self.OPERATOR_ARITY.get(op, 2)
            
            if arity == 1:
                child = self._generate_random_tree(depth + 1, method)
                return Node(NodeType.OPERATOR, op, [child])
            else:
                left = self._generate_random_tree(depth + 1, method)
                right = self._generate_random_tree(depth + 1, method)
                return Node(NodeType.OPERATOR, op, [left, right])
    
    def initialize_population(self) -> None:
        """Initialize population with diversity."""
        self.population = []
        
        # Mix of grow and full methods for diversity
        for i in range(self.population_size):
            if i < self.population_size // 2:
                tree = self._generate_random_tree(method='grow')
            else:
                tree = self._generate_random_tree(method='full')
            self.population.append(tree)
    
    def _evaluate_individual(self, individual: Node, x: np.ndarray, 
                           y_true: np.ndarray) -> Tuple[float, float, float]:
        """Evaluate individual fitness."""
        # Cache key for performance
        cache_key = (id(individual), x.tobytes(), y_true.tobytes())
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]
        
        try:
            y_pred = individual.evaluate_vectorized(x)
            
            # Check for invalid predictions
            valid_mask = ~np.isnan(y_pred) & ~np.isinf(y_pred)
            if np.sum(valid_mask) < len(x) * 0.9:  # Too many invalid values
                return -np.inf, np.inf, individual.size
            
            # Use only valid predictions
            if np.any(~valid_mask):
                y_pred_valid = y_pred[valid_mask]
                y_true_valid = y_true[valid_mask]
            else:
                y_pred_valid = y_pred
                y_true_valid = y_true
            
            # Mean squared error
            mse = np.mean((y_pred_valid - y_true_valid) ** 2)
            
            # Complexity
            complexity = individual.size + 0.3 * individual.depth
            
            # Fitness (negative because we maximize)
            fitness = -(mse + self.parsimony_coefficient * complexity)
            
            result = (fitness, mse, complexity)
            self._eval_cache[cache_key] = result
            return result
            
        except Exception as e:
            return -np.inf, np.inf, individual.size
    
    def evaluate_population(self, x: np.ndarray, y_true: np.ndarray) -> List[float]:
        """Evaluate entire population."""
        fitnesses = []
        for individual in self.population:
            fitness, _, _ = self._evaluate_individual(individual, x, y_true)
            fitnesses.append(fitness)
        return fitnesses
    
    def _select_node(self, tree: Node) -> Node:
        """Select node from tree for mutation/crossover."""
        # Collect all nodes
        nodes = []
        stack = [tree]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children)
        
        if not nodes:
            return tree
        
        # Prefer internal nodes (70% chance)
        internal_nodes = [node for node in nodes if node.children]
        if internal_nodes and random.random() < 0.7:
            return random.choice(internal_nodes)
        else:
            return random.choice(nodes)
    
    def mutate(self, individual: Node, mutation_rate: float = None) -> Node:
        """Apply mutation to individual."""
        if mutation_rate is None:
            mutation_rate = self.mutation_rate
        
        if random.random() > mutation_rate:
            return individual.copy()
        
        # Select node to mutate
        target = self._select_node(individual)
        
        # Generate replacement
        max_depth = self.max_depth
        if max_depth <= 0:
            # Terminal node
            if random.random() < 0.5:
                replacement = Node(NodeType.VARIABLE, 'x')
            else:
                replacement = Node(NodeType.CONSTANT, random.uniform(-2, 2))
        else:
            # Generate new subtree
            replacement = self._generate_random_tree(depth=0, method='grow')
        
        # Replace node
        return self._replace_node(individual, target, replacement)
    
    def _replace_node(self, tree: Node, target: Node, replacement: Node) -> Node:
        """Replace target node with replacement in tree."""
        if tree is target:
            return replacement.copy()
        
        new_children = []
        for child in tree.children:
            new_child = self._replace_node(child, target, replacement)
            new_children.append(new_child)
        
        return Node(tree.node_type, tree.value, new_children)
    
    def crossover(self, parent1: Node, parent2: Node) -> Tuple[Node, Node]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Select crossover points
        node1 = self._select_node(parent1)
        node2 = self._select_node(parent2)
        
        # Create children
        child1 = self._replace_node(parent1, node1, node2)
        child2 = self._replace_node(parent2, node2, node1)
        
        # Check depth constraints
        if child1.depth > self.max_depth or child2.depth > self.max_depth:
            return parent1.copy(), parent2.copy()
        
        return child1, child2
    
    def _optimize_constants(self, tree: Node, x: np.ndarray, 
                          y_true: np.ndarray) -> Node:
        """Optimize constants using gradient-based method."""
        constants = tree.collect_constants()
        if not constants:
            return tree.copy()
        
        # Extract current constants
        const_values = [float(c.value) for c in constants]
        n_constants = len(const_values)
        
        if n_constants == 0:
            return tree.copy()
        
        # Define objective function
        def objective(params):
            # Update constants in a copy
            tree_copy = tree.copy()
            const_nodes = tree_copy.collect_constants()
            for i, const in enumerate(const_nodes):
                const.value = params[i]
            
            # Evaluate
            y_pred = tree_copy.evaluate_vectorized(x)
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return np.inf
            
            mse = np.mean((y_pred - y_true) ** 2)
            return mse
        
        # Bounds for constants
        bounds = [(-10, 10)] * n_constants
        
        try:
            # Optimize using L-BFGS-B
            result = minimize(
                objective,
                const_values,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-6, 'disp': False}
            )
            
            if result.success:
                # Update with optimized values
                optimized_tree = tree.copy()
                opt_constants = optimized_tree.collect_constants()
                for i, const in enumerate(opt_constants):
                    const.value = result.x[i]
                return optimized_tree
        except Exception:
            pass
        
        return tree.copy()
    
    def tournament_select(self, fitnesses: List[float]) -> Node:
        """Select individual using tournament selection."""
        indices = np.random.choice(len(self.population), 
                                 size=min(self.tournament_size, len(self.population)), 
                                 replace=False)
        tournament_fitness = [fitnesses[i] for i in indices]
        best_idx = indices[np.argmax(tournament_fitness)]
        return self.population[best_idx].copy()
    
    def run_generation(self, x: np.ndarray, y_true: np.ndarray) -> Tuple[float, Node]:
        """Run one generation of evolution."""
        # Clear cache for new generation
        self._eval_cache.clear()
        
        # Evaluate population
        fitnesses = self.evaluate_population(x, y_true)
        
        # Sort by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]
        self.population = [self.population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]
        
        # Track best
        best_fitness = max(fitnesses)
        best_idx = np.argmax(fitnesses)
        best_individual = self.population[best_idx].copy()
        
        self.fitness_history.append(best_fitness)
        self.best_individuals.append(best_individual)
        
        # Track complexity
        avg_complexity = np.mean([ind.size for ind in self.population])
        self.complexity_history.append(avg_complexity)
        
        # Optimize constants in best individuals
        for i in range(min(3, len(self.population))):
            if fitnesses[i] > -np.inf:
                optimized = self._optimize_constants(self.population[i], x, y_true)
                opt_fitness, _, _ = self._evaluate_individual(optimized, x, y_true)
                if opt_fitness > fitnesses[i]:
                    self.population[i] = optimized
                    fitnesses[i] = opt_fitness
        
        # Create new generation
        new_population = []
        
        # Elitism: keep best individuals
        elite_indices = sorted_indices[:self.elite_size]
        new_population.extend([self.population[i].copy() for i in elite_indices])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(self.population) >= 2:
                parent1 = self.tournament_select(fitnesses)
                parent2 = self.tournament_select(fitnesses)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            else:
                parent = self.tournament_select(fitnesses)
                child = self.mutate(parent)
                new_population.append(child)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        return best_fitness, best_individual
    
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_true: np.ndarray = None,
        generations: int = 100,
        show_progress: bool = True,
        live_plot: bool = False
    ) -> Tuple[Node, List[float]]:
        """
        Run symbolic regression.
        
        Args:
            x: Input data
            y: Target data (noisy)
            y_true: True function values (for visualization)
            generations: Number of generations
            show_progress: Show progress bar
            live_plot: Show live visualization
        
        Returns:
            best_expression: Best discovered expression
            fitness_history: History of best fitness values
        """
        if y_true is None:
            y_true = y
        
        print("=" * 60)
        print("Symbolic Regression")
        print("=" * 60)
        print(f"Data points: {len(x)}")
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {generations}")
        print(f"Operators: {self.operators}")
        print("=" * 60)
        
        # Initialize
        self.initialize_population()
        self.fitness_history = []
        self.best_individuals = []
        self.complexity_history = []
        self.generation = 0
        
        # Early stopping
        best_fitness = -np.inf
        patience_counter = 0
        
        # Progress bar
        pbar = tqdm(range(generations), disable=not show_progress)
        
        for gen in pbar:
            fitness, best_indiv = self.run_generation(x, y_true)
            
            # Early stopping check
            if fitness > best_fitness:
                best_fitness = fitness
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.early_stopping_patience and patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping at generation {gen}")
                break
            
            # Update progress bar
            pbar.set_description(f"Gen {gen}: Fitness={fitness:.4f}")
            
            # Live plot
            if live_plot and gen % 10 == 0:
                self._plot_generation(x, y, y_true, gen)
        
        # Get best individual overall
        best_idx = np.argmax([self._evaluate_individual(ind, x, y_true)[0] 
                             for ind in self.best_individuals])
        best_expression = self.best_individuals[best_idx]
        
        # Final evaluation
        final_fitness, final_mse, final_complexity = self._evaluate_individual(best_expression, x, y_true)
        y_pred = best_expression.evaluate_vectorized(x)
        valid_mask = ~np.isnan(y_pred) & ~np.isinf(y_pred)
        if np.any(valid_mask):
            y_pred_valid = y_pred[valid_mask]
            y_true_valid = y_true[valid_mask]
            r2 = 1 - np.sum((y_true_valid - y_pred_valid) ** 2) / np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
        else:
            r2 = -np.inf
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Best expression: {best_expression}")
        print(f"Final fitness: {final_fitness:.4f}")
        print(f"MSE: {final_mse:.6f}")
        print(f"R² score: {r2:.4f}")
        print(f"Tree size: {best_expression.size} nodes")
        print(f"Tree depth: {best_expression.depth}")
        print("=" * 60)
        
        if live_plot:
            self._plot_final_results(x, y, y_true, best_expression)
            plt.show()
        
        return best_expression, self.fitness_history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using best individual."""
        if not self.best_individuals:
            raise ValueError("Model not fitted. Call fit() first.")
        
        best_idx = np.argmax([self._evaluate_individual(ind, x, np.zeros_like(x))[0] 
                             for ind in self.best_individuals])
        return self.best_individuals[best_idx].evaluate_vectorized(x)
    
    def _plot_generation(self, x: np.ndarray, y: np.ndarray, 
                        y_true: np.ndarray, generation: int) -> None:
        """Plot current generation status."""
        if not hasattr(self, 'fig') or self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        
        for ax in self.axes.flat:
            ax.clear()
        
        best = self.best_individuals[-1]
        y_pred = best.evaluate_vectorized(x)
        
        # Fitness history
        self.axes[0, 0].plot(self.fitness_history, 'b-', linewidth=2)
        self.axes[0, 0].set_title('Fitness Evolution')
        self.axes[0, 0].set_xlabel('Generation')
        self.axes[0, 0].set_ylabel('Fitness')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Data and fit
        self.axes[0, 1].scatter(x, y, alpha=0.3, color='gray', label='Noisy data', s=10)
        self.axes[0, 1].plot(x, y_true, 'g-', linewidth=2, label='True function')
        self.axes[0, 1].plot(x, y_pred, 'r--', linewidth=2, label='Best fit')
        self.axes[0, 1].set_title(f'Generation {generation}')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Complexity history
        self.axes[1, 0].plot(self.complexity_history, 'r-', linewidth=2, label='Avg Complexity')
        self.axes[1, 0].set_title('Complexity Evolution')
        self.axes[1, 0].set_xlabel('Generation')
        self.axes[1, 0].set_ylabel('Tree Size')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Population metrics
        if self.population:
            expr_lengths = [len(str(ind)) for ind in self.population[:20]]
            unique_expr = len(set(str(ind) for ind in self.population[:20]))
            avg_size = np.mean([ind.size for ind in self.population[:20]])
            
            metrics = ['Unique', 'Avg Size', 'Best Fitness']
            values = [unique_expr, avg_size, self.fitness_history[-1]]
            
            self.axes[1, 1].bar(metrics, values)
            self.axes[1, 1].set_title('Population Metrics')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def _plot_final_results(self, x: np.ndarray, y: np.ndarray,
                           y_true: np.ndarray, best: Node) -> None:
        """Plot final results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        y_pred = best.evaluate_vectorized(x)
        
        # Final fit
        axes[0, 0].scatter(x, y, alpha=0.3, color='gray', label='Noisy data', s=10)
        axes[0, 0].plot(x, y_true, 'g-', linewidth=2, label='True function')
        axes[0, 0].plot(x, y_pred, 'r--', linewidth=2, label='Discovered')
        axes[0, 0].set_title(f'Final Fit\n{str(best)}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_true - y_pred
        axes[0, 1].scatter(x, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Residuals')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Fitness history
        axes[1, 0].plot(self.fitness_history, 'b-', linewidth=2)
        axes[1, 0].set_title('Fitness History')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Fitness')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Complexity history
        axes[1, 1].plot(self.complexity_history, 'r-', linewidth=2)
        axes[1, 1].set_title('Complexity History')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Average Tree Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'best_individuals': self.best_individuals,
                'fitness_history': self.fitness_history,
                'complexity_history': self.complexity_history,
                'parameters': {
                    'operators': self.operators,
                    'max_depth': self.max_depth,
                    'population_size': self.population_size,
                    'parsimony_coefficient': self.parsimony_coefficient
                }
            }, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SymbolicRegressor':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        regressor = cls(**data['parameters'])
        regressor.best_individuals = data['best_individuals']
        regressor.fitness_history = data['fitness_history']
        regressor.complexity_history = data['complexity_history']
        
        return regressor
    
    def to_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """Convert best expression to callable function."""
        if not self.best_individuals:
            raise ValueError("No fitted model")
        
        best = self.best_individuals[-1]
        
        def func(x):
            if isinstance(x, (int, float)):
                return best.evaluate_vectorized(np.array([x]))[0]
            else:
                return best.evaluate_vectorized(np.array(x))
        
        return func

def create_example_data():
    """Create example data for demonstration."""
    np.random.seed(42)
    x = np.linspace(-5, 5, 100)
    y_true = np.sin(x) * x + 0.5 * np.exp(-0.1 * x**2)
    y_data = y_true + np.random.normal(0, 0.15, len(x))
    return x, y_data, y_true

def main():
    """Main function to run example."""
    # Create data
    x, y_data, y_true = create_example_data()
    
    # Create and run regressor
    sr = SymbolicRegressor(
        operators=['+', '-', '*', '/', 'sin', 'exp', 'cos'],
        max_depth=6,
        population_size=150,
        parsimony_coefficient=0.005,
        early_stopping_patience=25,
        random_state=42
    )
    
    # Run evolution
    best_expr, history = sr.fit(
        x, y_data, y_true,
        generations=80,
        show_progress=True,
        live_plot=False
    )
    
    # Test prediction
    x_test = np.array([-2, -1, 0, 1, 2])
    y_test = sr.predict(x_test)
    print(f"\nTest predictions:")
    for xi, yi in zip(x_test, y_test):
        print(f"  f({xi}) = {yi:.4f}")
    
    # Save model
    sr.save_model("symbolic_model.pkl")
    print("\nModel saved to 'symbolic_model.pkl'")
    
    # Load and test
    sr_loaded = SymbolicRegressor.load_model("symbolic_model.pkl")
    func = sr_loaded.to_function()
    print(f"\nLoaded function: f(2.5) = {func(2.5):.4f}")
    
    # Plot results
    sr._plot_final_results(x, y_data, y_true, best_expr)
    plt.show()
    
    return sr

if __name__ == "__main__":
    print("Symbolic Regression System")
    print("=" * 60)
    
    try:
        sr = main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()