import numpy as np

def rk4_complete(f, x0, y0, x_target, h=0.1):

    print(f"\n{'='*60}")
    print("RUNGE-KUTTA 4 (RK4) METHOD")
    print(f"{'='*60}")
    print(f"ODE: y' = f(x,y)")
    print(f"Initial: y({x0}) = {y0}")
    print(f"Target:  y({x_target}) with step size h = {h}")
    
    x = x0
    y = y0
    step = 0
    
    print(f"\nStep {step}: Start at (x={x}, y={y})")
    
    while x < x_target:
        # Adjust last step if needed
        if x + h > x_target:
            h_adj = x_target - x
        else:
            h_adj = h
        
        # RK4's 4 slope estimates
        k1 = f(x, y)
        k2 = f(x + h_adj/2, y + h_adj*k1/2)
        k3 = f(x + h_adj/2, y + h_adj*k2/2)
        k4 = f(x + h_adj, y + h_adj*k3)
        
        step += 1
        print(f"\nStep {step}:")
        print(f"  k1 = f({x:.2f}, {y:.4f}) = {k1:.4f}")
        print(f"  k2 = f({x+h_adj/2:.2f}, {y:.4f}+{h_adj}×{k1:.4f}/2) = {k2:.4f}")
        print(f"  k3 = f({x+h_adj/2:.2f}, {y:.4f}+{h_adj}×{k2:.4f}/2) = {k3:.4f}")
        print(f"  k4 = f({x+h_adj:.2f}, {y:.4f}+{h_adj}×{k3:.4f}) = {k4:.4f}")
        
        # The ÷6 comes from weights: k1(1), k2(2), k3(2), k4(1) → total=6
        weighted_avg = (k1 + 2*k2 + 2*k3 + k4) / 6
        y_new = y + h_adj * weighted_avg
        
        print(f"  Weighted average = (k1 + 2k2 + 2k3 + k4)/6")
        print(f"                   = ({k1:.4f} + 2×{k2:.4f} + 2×{k3:.4f} + {k4:.4f})/6")
        print(f"                   = {(k1 + 2*k2 + 2*k3 + k4):.4f}/6 = {weighted_avg:.4f}")
        print(f"  y_new = {y:.4f} + {h_adj} × {weighted_avg:.4f} = {y_new:.4f}")
        
        x = x + h_adj
        y = y_new
    
    print(f"\nFinal result: y({x_target}) ≈ {y:.6f}")
    return y

# ===== EXAMPLE 1: Exponential Growth =====
def f_exponential(x, y):
    """f(x,y) = y (for dy/dx = y)"""
    return y

print("\nEXAMPLE 1: dy/dx = y, y(0) = 1")
print("Find y(1) = e ≈ 2.71828")

result1 = rk4_complete(f_exponential, 0, 1, 1.0, h=0.5)

# Compare with exact
exact1 = np.exp(1)
print(f"\nComparison:")
print(f"RK4 approximation: y(1) ≈ {result1:.6f}")
print(f"Exact value (e):    y(1) = {exact1:.6f}")
print(f"Error:             {abs(exact1 - result1):.6f}")
print(f"Percentage error:  {abs(exact1 - result1)/exact1*100:.4f}%")

# ===== EXAMPLE 2: Another ODE =====
def f_xy(x, y):
    """f(x,y) = x + y (for dy/dx = x + y)"""
    return x + y

print(f"\n{'='*60}")
print("EXAMPLE 2: dy/dx = x + y, y(0) = 1")
print("Find y(0.5)")

result2 = rk4_complete(f_xy, 0, 1, 0.5, h=0.1)

# Exact solution for y' = x + y, y(0) = 1
exact2 = -0.5 - 1 + 2*np.exp(0.5)
print(f"\nComparison:")
print(f"RK4 approximation: y(0.5) ≈ {result2:.6f}")
print(f"Exact value:       y(0.5) = {exact2:.6f}")
print(f"Error:             {abs(exact2 - result2):.6f}")

# ===== MINIMAL VERSION (Just the math) =====
def rk4_minimal(f, x0, y0, x_target, h=0.1):
    """Minimal RK4 - just the calculations"""
    x, y = x0, y0
    
    while x < x_target:
        if x + h > x_target:
            h = x_target - x
        
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        
        y = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6  # The ÷6 is here!
        x = x + h
    
    return y

print(f"\n{'='*60}")
print("MINIMAL VERSION TEST")
print(f"{'='*60}")

# Quick test
result_minimal = rk4_minimal(f_exponential, 0, 1, 1.0, h=0.5)
print(f"Minimal RK4: y(1) ≈ {result_minimal:.6f}")
print(f"Exact:       y(1) = {exact1:.6f}")

# ===== COMPARE DIFFERENT STEP SIZES =====
print(f"\n{'='*60}")
print("RK4 WITH DIFFERENT STEP SIZES")
print(f"{'='*60}")

print(f"\nProblem: y' = y, y(0) = 1, find y(1)")
print(f"Exact: e = {exact1:.6f}\n")

step_sizes = [0.5, 0.25, 0.1, 0.05]
print(f"{'Step size':<10} {'Result':<15} {'Error':<15} {'% Error':<10}")
print(f"{'-'*50}")

for h_size in step_sizes:
    result = rk4_minimal(f_exponential, 0, 1, 1.0, h=h_size)
    error = abs(exact1 - result)
    percent = error / exact1 * 100
    print(f"{h_size:<10} {result:<15.6f} {error:<15.6f} {percent:<10.4f}%")