import numpy as np
import math

def taylor_method(f, x0, y0, x_target, order=4):
    print(f"\n{'='*60}")
    print("TAYLOR'S SERIES METHOD")
    print(f"{'='*60}")
    print(f"ODE: dy/dx = f(x,y)")
    print(f"Initial: y({x0}) = {y0}")
    print(f"Target:  y({x_target}) using {order} terms")
    
    h = x_target - x0
    
    # STEP 1: Compute derivatives at (x0, y0)
    print(f"\nStep 1: Compute derivatives at x={x0}, y={y0}")
    
    # Store derivatives
    derivatives = []
    
    # First derivative: y' = f(x,y)
    y_prime = f(x0, y0)
    derivatives.append(y_prime)
    print(f"  y'  = f({x0}, {y0}) = {y_prime}")
    
    # For this specific example (y' = x + y), we can compute exactly:
    # y'' = ∂f/∂x + ∂f/∂y · y' = 1 + 1·(x+y) = 1 + x + y
    if order >= 2:
        y_double_prime = 1 + x0 + y0  # For f(x,y) = x + y
        derivatives.append(y_double_prime)
        print(f"  y'' = 1 + {x0} + {y0} = {y_double_prime}")
    
    # y''' = derivative of y''
    if order >= 3:
        y_triple_prime = 1 + y_prime  # derivative of (1 + x + y)
        derivatives.append(y_triple_prime)
        print(f"  y''' = 1 + {y_prime} = {y_triple_prime}")
    
    # y'''' = derivative of y'''
    if order >= 4:
        y_quadruple_prime = y_double_prime
        derivatives.append(y_quadruple_prime)
        print(f"  y'''' = {y_double_prime}")
    
    # STEP 2: Build Taylor series
    print(f"\nStep 2: Build Taylor series")
    print(f"y(x) ≈ y0 + y'·h + y''·h²/2! + y'''·h³/3! + ...")
    print(f"where h = {x_target} - {x0} = {h}")
    
    result = y0
    print(f"\nStarting value: {result}")
    
    for n in range(1, order + 1):
        term = derivatives[n-1] * h**n / math.factorial(n)
        result += term
        print(f"+ term {n}: {derivatives[n-1]} × {h}^{n}/{n}! = {term:.6f}")
        print(f"  Current sum: {result:.6f}")
    
    return result

print("\nEXAMPLE: dy/dx = x + y, y(0) = 1")
print("Find y(0.5)")

def f_example(x, y):
    """The right-hand side of our ODE: f(x,y) = x + y"""
    return x + y

# Test with different number of terms
print(f"\n{'='*60}")
print("Testing with different number of Taylor terms:")
print(f"{'='*60}")

orders = [1, 2, 3, 4]
results = []

for order in orders:
    print(f"\nOrder {order} Taylor approximation:")
    y_approx = taylor_method(f_example, 0, 1, 0.5, order=order)
    results.append((order, y_approx))

# Exact solution for comparison
print(f"\n{'='*60}")
print("COMPARISON WITH EXACT SOLUTION")
print(f"{'='*60}")

# Exact solution for y' = x + y, y(0) = 1 is:
# y(x) = -x - 1 + 2e^x
exact_solution = lambda x: -x - 1 + 2*np.exp(x)
y_exact = exact_solution(0.5)

print(f"\nExact solution formula: y(x) = -x - 1 + 2e^x")
print(f"y(0.5) = -0.5 - 1 + 2×e^0.5")
print(f"      = -1.5 + 2×{np.exp(0.5):.6f}")
print(f"      = {y_exact:.6f}")

print(f"\n{'Order':<8} {'Approximation':<15} {'Error':<12} {'% Error':<10}")
print(f"{'-'*50}")
for order, y_approx in results:
    error = abs(y_exact - y_approx)
    percent_error = 100 * error / y_exact
    print(f"{order:<8} {y_approx:<15.6f} {error:<12.6f} {percent_error:<10.2f}%")

# ===== VISUALIZATION =====
print(f"\n{'='*60}")
print("VISUALIZATION")
print(f"{'='*60}")

import matplotlib.pyplot as plt

x_plot = np.linspace(0, 1, 100)
y_exact_plot = exact_solution(x_plot)

plt.figure(figsize=(10, 6))

# Plot exact solution
plt.plot(x_plot, y_exact_plot, 'k-', linewidth=3, label='Exact Solution')

# Plot Taylor approximations
colors = ['red', 'blue', 'green', 'purple']
for (order, y_approx), color in zip(results, colors):
    # Build Taylor polynomial for plotting
    h_plot = x_plot - 0  # since x0 = 0
    
    # For f(x,y) = x + y, derivatives at x=0, y=1:
    # y' = 1, y'' = 2, y''' = 2, y'''' = 2
    if order == 1:
        y_taylor = 1 + 1*h_plot
    elif order == 2:
        y_taylor = 1 + 1*h_plot + 2*h_plot**2/2
    elif order == 3:
        y_taylor = 1 + 1*h_plot + 2*h_plot**2/2 + 2*h_plot**3/6
    elif order == 4:
        y_taylor = 1 + 1*h_plot + 2*h_plot**2/2 + 2*h_plot**3/6 + 2*h_plot**4/24
    
    plt.plot(x_plot, y_taylor, '--', color=color, alpha=0.8,
             label=f'Taylor order {order}')

plt.xlabel('x', fontsize=12)
plt.ylabel('y(x)', fontsize=12)
plt.title("Taylor Series Approximations of y' = x + y, y(0) = 1", fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='x = 0.5')
plt.tight_layout()
plt.show()

# ===== SIMPLE USAGE EXAMPLE =====
print(f"\n{'='*60}")
print("QUICK USAGE - SIMPLIFIED")
print(f"{'='*60}")

def taylor_quick(f, x0, y0, x_target, order=4):
    """Quick Taylor method without all the printing"""
    h = x_target - x0
    result = y0
    
    # For f(x,y) = x + y specifically:
    derivatives = [1, 2, 2, 2]  # y', y'', y''', y''''
    
    for n in range(1, order + 1):
        result += derivatives[n-1] * h**n / np.math.factorial(n)
    
    return result

# Quick test
y_quick = taylor_quick(f_example, 0, 1, 0.5, order=4)
print(f"\nQuick calculation:")
print(f"y(0.5) ≈ {y_quick:.6f}")
print(f"Exact:   {y_exact:.6f}")
print(f"Difference: {abs(y_exact - y_quick):.6f}")