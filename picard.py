import numpy as np

def f1(x, y):
    return x + y

def picard(f, x0, y0, x_target, iterations=3, n_points=100):
    
    # Create x values for integration
    x_points = np.linspace(x0, x_target, n_points)
    
    # Start with initial guess: y(x) = y0 (constant)
    y_prev = y0 * np.ones_like(x_points)
    
    for i in range(iterations):
        print(f"\n--- Iteration {i+1} ---")
        
        # Compute new approximation: y_new(x) = y0 + ∫ f(t, y_prev(t)) dt
        y_new = np.zeros_like(x_points)
        y_new[0] = y0  # Initial condition
        
        # For each x, integrate from x0 to x using trapezoidal rule
        for j in range(1, len(x_points)):
            x = x_points[j]
            # Integration points from x0 to x
            t_vals = np.linspace(x0, x, j+1)
            # Get y_prev values at these points
            y_prev_vals = np.interp(t_vals, x_points[:j+1], y_prev[:j+1])
            # Integrate f(t, y_prev(t))
            integrand = f(t_vals, y_prev_vals)
            integral = np.trapz(integrand, t_vals)
            y_new[j] = y0 + integral
        
        # Update for next iteration
        y_prev = y_new.copy()
        
        # Print result at target
        print(f"  y({x_target}) ≈ {y_new[-1]:.6f}")
    
    return y_new[-1]

print("\n" + "="*60)
better_result = picard(f1, 0, 1, 0.5, iterations=3)
print(f"\nBetter approximation: y(0.5) ≈ {better_result:.6f}")
print(f"Exact solution:        y(0.5) ≈ 1.797443")
