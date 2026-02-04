import numpy as np

def euler_simple(f, x0, y0, x_target, h=0.1):
    """
    SIMPLEST Euler's method
    Formula: y_new = y_old + h × f(x_old, y_old)
    """
    print(f"\nEULER'S METHOD")
    print(f"Solving: y' = f(x,y), y({x0}) = {y0}")
    print(f"Finding y({x_target}) with h = {h}")
    
    x = x0
    y = y0
    step = 0
    
    print(f"\nStep {step}: Start at (x={x}, y={y})")
    
    #keep going till we reach target x
    while x < x_target:
        # if overstepping it makes the steps smaller 
        if x + h > x_target:
            h = x_target - x
        
        # Calculate slope at current point
        slope = f(x, y)
        
        # Euler update(core formula)
        y_new = y + h * slope
        x_new = x + h
        
        step += 1
        print(f"Step {step}:")
        print(f"  At (x={x:.2f}, y={y:.4f}), slope = f({x:.2f}, {y:.4f}) = {slope:.4f}")
        print(f"  y_new = {y:.4f} + {h} × {slope:.4f} = {y_new:.4f}")
        print(f"  Move to (x={x_new:.2f}, y={y_new:.4f})")
        
        #update position of x and y
        x, y = x_new, y_new
    
    print(f"\nResult: y({x_target}) ≈ {y:.6f}")
    return y

def f_example(x, y):
    """f(x,y) = y (for dy/dx = y)"""
    return y

print("=" * 50)
print("EXAMPLE: dy/dx = y, y(0) = 1")
print("Find y(1) = e ≈ 2.71828")
print("=" * 50)

result = euler_simple(f_example, 0, 1, 1.0, h=0.5)

# Compare with exact
exact = np.exp(1)
print(f"\nExact value: e = {exact:.6f}")
print(f"Error: {abs(exact - result):.6f}")