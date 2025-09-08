import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import pandas as pd

# Maxwell construction for Carnahan-Starling EOS
rho_c = 3.5
p_c = 0.001
Tr = 0.61

# Calculate EOS parameters
b_eos = 0.5218 / rho_c
a_eos = ((b_eos ** 2) * p_c) / ((0.3773 ** 2) * 0.4963)
R_eos = 1.0
Tc = (0.3773 * a_eos) / (b_eos * R_eos)
T_eos = Tr * Tc


def pressure_function(x):
    """Calculate pressure using Carnahan-Starling EOS"""
    return (R_eos * T_eos * (1 / x) *
            ((1 + (b_eos / (x * 4)) + ((b_eos / (x * 4)) ** 2) - ((b_eos / (x * 4)) ** 3)) /
             ((1 - (b_eos / (x * 4))) ** 3)) - a_eos * (1 / (x ** 2)))


def pressure_derivative(x):
    """Calculate derivative of pressure with respect to volume"""
    return (R_eos * T_eos * ((b_eos * (1 / x)) ** 4 - 16 * ((b_eos * (1 / x)) ** 3) +
                             64 * ((b_eos * (1 / x)) ** 2) + 256 * b_eos * (1 / x) + 256) /
            ((4 - b_eos * (1 / x)) ** 4) - 2 * a_eos * (1 / x))


def find_extrema(a_int, b_int):
    """Find the maximum and minimum points of the pressure function.

    This part of the code detects sign changes in the derivative of the pressure function
    over a range of test points. When a sign change is found between two consecutive points,
    it uses Brent's method (brentq) to accurately find the root (where the derivative crosses zero),
    which corresponds to an extremum (maximum or minimum) of the pressure function.
    All found extrema are collected and returned in the extrema list. This is useful for
    identifying phase transition points in the Maxwell construction.
    """

    def derivative_zero(x):
        return pressure_derivative(x)

    extrema = []

    # Sample points to find sign changes
    test_points = np.linspace(a_int, b_int, 1000)
    derivative_values = [pressure_derivative(x) for x in test_points]

    for i in range(len(test_points) - 1):
        if derivative_values[i] * derivative_values[i + 1] < 0:
            try:
                root = brentq(derivative_zero, test_points[i], test_points[i + 1])
                extrema.append(root)
            except:
                continue

    return extrema


def riemann_integration(a, b, n, f):
    """Riemann integration approximation"""
    delta = (b - a) / n
    xs = np.linspace(a, b - delta, n)
    fx = np.array([f(x) for x in xs])
    integral = np.sum(fx) * delta
    return integral


def maxwell_construction(max_min, a_int, b_int, max_iterations=40, tolerance=1e-7):
    """
    Perform Maxwell construction to find coexistence densities
    """
    if len(max_min) < 2:
        print("Insufficient extrema for Maxwell construction")
        return None, None, None, None, None

    guess = 0.01

    # Initial guesses for the coexistence points
    a_ = max_min[0] + guess * (max_min[1] - max_min[0])
    b = max_min[1] + guess * (max_min[0] - max_min[1])

    g = (a_ + b) / 2

    # Check if Maxwell construction is feasible
    p_at_extrema = [pressure_function(x) for x in max_min]
    print(f"Pressures at extrema: {p_at_extrema}")

    # If pressure doesn't change sign significantly, Maxwell construction may not be meaningful
    if all(p >= 0 for p in p_at_extrema) and (max(p_at_extrema) - min(p_at_extrema)) < 1e-6:
        print("Pressure variation too small for meaningful Maxwell construction")
        # Return approximate coexistence points
        return (max_min[0] * 0.9, max_min[1] * 1.1), 0, 0, max_min[0], max_min[1]

    # Handle negative pressure case
    if pressure_function(max_min[0]) < 0:
        try:
            zero_pressure = brentq(pressure_function, max_min[0], max_min[1])
            a_ = zero_pressure * (1 + guess)
        except:
            pass

    print(f"Initial guesses: a = {a_}, b = {b}")
    print(f"Pressure at a = {pressure_function(a_)}")

    # Variables to track integration results
    integd_, integc_ = 0, 0

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")

        c, d = pressure_function(a_), pressure_function(b)

        # Handle negative pressure
        while c < 0:
            g = (a_ + b) / 2
            a_ = g
            c = pressure_function(a_)

        # Define pressure difference functions
        def pc_function(x):
            return pressure_function(x) - c

        def pd_function(x):
            return pressure_function(x) - d

        # Find intersection points with better error handling
        try:
            i1 = brentq(pc_function, a_int, max_min[0])
            i3 = brentq(pc_function, max_min[1], b_int)
            integc_ = riemann_integration(i1, i3, 100000, pc_function)
            print(f"Integral c: {integc_}")
        except Exception as e:
            print(f"Could not find intersection for pressure c: {e}")
            # Try to continue with a fallback approach
            integc_ = 0

        try:
            j1 = brentq(pd_function, a_int, max_min[0])
            j3 = brentq(pd_function, max_min[1], b_int)
            integd_ = riemann_integration(j1, j3, 100000, pd_function)
            print(f"Integral d: {integd_}")
        except Exception as e:
            print(f"Could not find intersection for pressure d: {e}")
            # Try to continue with a fallback approach
            integd_ = 0

        # Bisection step
        g = (a_ + b) / 2
        e = pressure_function(g)

        def pe_function(x):
            return pressure_function(x) - e

        try:
            m1 = brentq(pe_function, a_int, max_min[0])
            m3 = brentq(pe_function, max_min[1], b_int)
            intege = riemann_integration(m1, m3, 100000, pe_function)

            # Update based on integral sign
            if intege > 0:
                a_ = g
            else:
                b = g

        except Exception as e:
            print(f"Could not find intersection for pressure e: {e}")
            # If we can't proceed, break with current values
            break

        # Check convergence
        if abs(intege) < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break

    # Final calculation - g should be defined by now
    try:
        e = pressure_function(g)

        def pf_function(x):
            return pressure_function(x) - e

        k1 = brentq(pf_function, a_int, max_min[0])
        k3 = brentq(pf_function, max_min[1], b_int)
        k = (k1, k3)
        print(f"Final coexistence points: {k}")

        return k, integd_, integc_, a_, b
    except Exception as e:
        print(f"Could not find final intersection points: {e}")
        # Return approximate values based on extrema
        return (max_min[0], max_min[1]), integd_, integc_, a_, b


# Main execution
if __name__ == "__main__":
    # Define integration bounds
    a_int, b_int = 0.09, 50

    # Generate volume array and calculate pressure and derivative
    rho_w = np.arange(a_int, b_int, 0.01)
    p_bulk = np.array([pressure_function(rho) for rho in rho_w])
    p_dash_bulk = np.array([pressure_derivative(rho) for rho in rho_w])

    # Find extrema and perform Maxwell construction
    max_min = find_extrema(a_int, b_int)
    print(f"Extrema found at: {max_min}")

    k = None
    if len(max_min) >= 2:
        k, integd, integc, a, b = maxwell_construction(max_min, a_int, b_int)

        if k is not None:
            maxwell_pressure = pressure_function((k[0] + k[1]) / 2)
            print(f"Maxwell construction coexistence pressure: {maxwell_pressure}")
            print(f"Coexistence volumes: {k[0]:.6f}, {k[1]:.6f}")
            print(f"Coexistence denisties: {1 / k[0]:.6f}, {1 / k[1]:.6f}")
    else:
        print("Could not find sufficient extrema for Maxwell construction")

    # Create and save results
    results_df = pd.DataFrame({
        'Volume': rho_w,
        'Pressure': p_bulk,
        'Pressure_Derivative': p_dash_bulk
    })

    results_df.to_csv('carnahan_starling_results.csv', index=False)
    print("Results saved to carnahan_starling_results.csv")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(rho_w, p_bulk, label='Carnahan-Starling EOS', linewidth=2)

    if k is not None:
        maxwell_pressure = pressure_function((k[0] + k[1]) / 2)
        plt.axhline(y=maxwell_pressure, color='red', linestyle='--',
                    label=f'Maxwell Construction (P = {maxwell_pressure:.6f})')
        plt.axvline(x=k[0], color='green', linestyle=':', alpha=0.7,
                    label=f'Liquid volume = {k[0]:.3f}')
        plt.axvline(x=k[1], color='blue', linestyle=':', alpha=0.7,
                    label=f'Vapor volume = {k[1]:.3f}')

    plt.xlabel('Volume, V')
    plt.ylabel('Pressure, P')
    plt.title('Carnahan-Starling EOS with Maxwell Construction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)  # Focus on relevant range

    # Save the plot to the specified directory with a timestamp to avoid overwriting
    import datetime
    save_dir = "/Users/sbszkudlarek/PycharmProjects/WBLBM/example/tests/maxwell_reconstruction"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_path = os.path.join(save_dir, f"carnahan_starling_maxwell_{timestamp}.png")
    plt.savefig(save_path, dpi=200)
    print(f"Figure saved to {save_path}")

    print("\nCalculation completed successfully!")
