def modified_euler_simple(f, x0, y0, x_target, h=0.1):

    print(f"\nMODIFIED EULER'S METHOD")
    print(f"Solving: y' = f(x,y), y({x0}) = {y0}")
    print(f"Finding y({x_target}) with h = {h}")
    
    x = x0
    y = y0
    step = 0
    
    print(f"\nStep {step}: Start at (x={x}, y={y})")
    
    while x < x_target:
        # Adjust last step if needed
        if x + h > x_target:
            h = x_target - x
        
        # Step 1: Predictor (regular Euler)
        slope1 = f(x, y)
        y_predict = y + h * slope1
        
        # Step 2: Corrector (average slopes)
        slope2 = f(x + h, y_predict)
        y_corrected = y + h * (slope1 + slope2) / 2
        
        step += 1
        print(f"\nStep {step}:")
        print(f"  1. Predictor (Euler):")
        print(f"     slope1 = f({x:.2f}, {y:.4f}) = {slope1:.4f}")
        print(f"     y_predict = {y:.4f} + {h}×{slope1:.4f} = {y_predict:.4f}")
        print(f"  2. Corrector (average):")
        print(f"     slope2 = f({x+h:.2f}, {y_predict:.4f}) = {slope2:.4f}")
        print(f"     Average slope = ({slope1:.4f} + {slope2:.4f})/2 = {(slope1+slope2)/2:.4f}")
        print(f"     y_corrected = {y:.4f} + {h}×{(slope1+slope2)/2:.4f} = {y_corrected:.4f}")
        
        # Update
        x = x + h
        y = y_corrected
    
    print(f"\nResult: y({x_target}) ≈ {y:.6f}")
    return y

def f_example(x, y):
    """f(x,y) = y (for dy/dx = y)"""
    return y

print("=" * 60)
print("EXAMPLE: dy/dx = y, y(0) = 1")
print("Find y(1) = e ≈ 2.71828")
print("=" * 60)

result = modified_euler_simple(f_example, 0, 1, 1.0, h=0.5)

# Compare with exact
import numpy as np
exact = np.exp(1)
print(f"\nExact value: e = {exact:.6f}")
print(f"Error: {abs(exact - result):.6f}")