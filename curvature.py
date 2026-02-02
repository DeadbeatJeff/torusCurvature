import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def dh_htm(theta1, theta2, r, d):
    """Computes the standard DH Homogeneous Transformation Matrix."""
    return sp.Matrix([
        [sp.cos(theta1), -sp.sin(theta1)*sp.cos(theta2),  sp.sin(theta1)*sp.sin(theta2), r*sp.cos(theta1)],
        [sp.sin(theta1),  sp.cos(theta1)*sp.cos(theta2), -sp.cos(theta1)*sp.sin(theta2), r*sp.sin(theta1)],
        [0,              sp.sin(theta2),               sp.cos(theta2),              d],
        [0,              0,                           0,                          1]
    ])

# 1. Initialize Symbolic Variables
g = sp.symbols('g')
n_joints = 2 # Based on dhParams in curvature.m

# Joint variables: theta, d_theta, dd_theta
Theta = sp.symbols(f'Theta1:{n_joints+1}')
thetaDot = sp.symbols(f'thetaDot1:{n_joints+1}')
doubleThetaDot = sp.symbols(f'doubleThetaDot1:{n_joints+1}')

# Physical parameters: mass, inertia
m = sp.symbols(f'm1:{n_joints+1}')
Izz = sp.symbols(f'Izz1:{n_joints+1}')
L = sp.symbols(f'L1:{n_joints+1}')

# 2. Define DH Parameters [theta, alpha, r, d]
# Based on curvature.m: [0 0 1 0; 0 0 0 1]
dh_params = [
    [Theta[0], 0, 1, 0],
    [Theta[1], 0, 0, 1]
]

# 3. Compute Transformations
T = []
for params in dh_params:
    T.append(dh_htm(*params))

    # 2. Kinematics (Center of Mass positions)
x = [0] * n_joints
y = [0] * n_joints

# Link 1 CoM
x[0] = (1/2) * L[0] * sp.cos(Theta[0])
y[0] = (1/2) * L[0] * sp.sin(Theta[0])

# Link 2 CoM (Simplified planar geometry logic from your MATLAB loop)
# Position of joint 2 + half of link 2
x[1] = L[0] * sp.cos(Theta[0]) + (1/2) * L[1] * sp.cos(Theta[0] + Theta[1])
y[1] = L[0] * sp.sin(Theta[0]) + (1/2) * L[1] * sp.sin(Theta[0] + Theta[1])

# 4. Velocities
xdot = [sp.diff(pos, t) for pos, t in zip(x, Theta)] # Partial diffs handled via Chain Rule below
v_sq = []
for i in range(n_joints):
    # Total derivative: dx/dt = sum( (dx/dTheta_j) * thetaDot_j )
    xd = sum(sp.diff(x[i], Theta[j]) * thetaDot[j] for j in range(n_joints))
    yd = sum(sp.diff(y[i], Theta[j]) * thetaDot[j] for j in range(n_joints))
    v_sq.append(xd**2 + yd**2)

# 5. Energy Calculations
KE = 0
V = 0
thetaDotSum = 0

for i in range(n_joints):
    thetaDotSum += thetaDot[i]
    # Translational + Rotational Kinetic Energy
    KE += (1/2) * m[i] * v_sq[i] + (1/2) * Izz[i] * (thetaDotSum)**2
    # Potential Energy (Gravity)
    V += m[i] * g * y[i]

Lagrangian = KE - V

# 6. Equations of Motion (Euler-Lagrange)
# d/dt(dL/dthetaDot) - dL/dtheta = Tau
RHS = []
for i in range(n_joints):
    dL_dthetaDot = sp.diff(Lagrangian, thetaDot[i])
    
    # Time derivative of dL/dthetaDot
    term1 = 0
    for j in range(n_joints):
        term1 += sp.diff(dL_dthetaDot, Theta[j]) * thetaDot[j]
        term1 += sp.diff(dL_dthetaDot, thetaDot[j]) * doubleThetaDot[j]
        
    term2 = sp.diff(Lagrangian, Theta[i])
    RHS.append(sp.simplify(term1 - term2))

# 7. Create a dictionary of numerical values
# This replaces the MATLAB loop for assigning constants
rob_values = {}

for i in range(n_joints):
    # Map the symbolic variable to the numeric value
    rob_values[m[i]] = 0.181
    rob_values[L[i]] = 0.250
    rob_values[Izz[i]] = 0.001
    # If you included the full inertia tensor in your EoM:
    # rob_values[Ixx[i]] = 0.093805
    # rob_values[Iyy[i]] = 0.001

# 8. Add gravity if needed
rob_values[g] = 9.81

# 9. Extract Mass Matrix (MJeff)
# The Mass Matrix M is the coefficient of doubleThetaDot
MJeff = sp.Matrix.zeros(n_joints, n_joints)
for i in range(n_joints):
    for j in range(n_joints):
        # Collect coefficients of the acceleration terms
        MJeff[i, j] = sp.simplify(sp.diff(RHS[i], doubleThetaDot[j]))

# 10. Substitute values into your Mass Matrix (MJeff) or Lagrangian
# This creates a version of the matrix that only depends on Theta
M_numeric_sym = MJeff.subs(rob_values)

print("Mass Matrix (Riemannian Metric):")
sp.pprint(sp.simplify(M_numeric_sym).evalf(3))

# 11. Compute the Inverse Mass Matrix (MJeffinv)
# MJeff is the Riemannian metric tensor g_ij
MJeffinv = MJeff.inv()

# 12. Compute Christoffel Symbols of the First Kind (Gamma1st)
# Gamma_ijk = 0.5 * ( d/dq_k(g_ij) + d/dq_j(g_ik) - d/dq_i(g_jk) )
Gamma1st = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)

for i in range(n_joints):
    for j in range(n_joints):
        for k in range(n_joints):
            term = 0.5 * (
                sp.diff(MJeff[i, j], Theta[k]) + 
                sp.diff(MJeff[i, k], Theta[j]) - 
                sp.diff(MJeff[j, k], Theta[i])
            )
            Gamma1st[i, j, k] = sp.simplify(term)

# 13. Compute Christoffel Symbols of the Second Kind (Gamma2nd)
# Gamma^i_jk = sum_l ( g^il * Gamma_ljk )
Gamma2nd = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints)

for i in range(n_joints):
    for j in range(n_joints):
        for k in range(n_joints):
            gamma_val = 0
            for l in range(n_joints):
                gamma_val += MJeffinv[i, l] * Gamma1st[l, j, k]
            Gamma2nd[i, j, k] = sp.simplify(gamma_val)

print("Christoffel symbols of the second kind computed.")

# 14. Symbolic Riemannian Curvature Tensor Calculation
# Define the Riemann computation function before use
def compute_riemann(Gamma2nd, Theta, n):
    Riemann = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    # Derivative terms: diff(Gamma_ijl, Theta_k) - diff(Gamma_ijk, Theta_l)
                    term1 = sp.diff(Gamma2nd[i, j, l], Theta[k])
                    term2 = sp.diff(Gamma2nd[i, j, k], Theta[l])
                    
                    # Summation term for indices p=1 to n
                    sum_term = 0
                    for p in range(n):
                        sum_term += (Gamma2nd[i, k, p] * Gamma2nd[p, j, l] - 
                                     Gamma2nd[i, l, p] * Gamma2nd[p, j, k])
                    
                    Riemann[i, j, k, l] = term1 - term2 + sum_term
    return Riemann

# Note: The original MATLAB script calculates Christoffel symbols (Gamma2nd)
# and then the Riemann tensor.
n = n_joints
RiemannContra = sp.MutableDenseNDimArray.zeros(n, n, n, n)
RiemannContra = compute_riemann(Gamma2nd, Theta, n)

print("Contravariant Riemann Tensor structure computed.")

# 15. Compute RiemannCovar (Fully Covariant Riemann Tensor)
# R_{lijk} = sum_m ( g_{lm} * R^m_{ijk} )
RiemannCovar = sp.MutableDenseNDimArray.zeros(n_joints, n_joints, n_joints, n_joints)

for i in range(n_joints):
    for j in range(n_joints):
        for k in range(n_joints):
            for l in range(n_joints):
                sum_term = 0
                for m_idx in range(n_joints):
                    # RiemannContra was computed in the previous step
                    sum_term += MJeff[l, m_idx] * RiemannContra[m_idx, i, j, k]
                RiemannCovar[l, i, j, k] = sp.simplify(sum_term)

print("Fully Covariant Riemann Tensor structure computed.")

# 16. Compute Gaussian Curvature (K)
# For a 2D surface, K = R_{1212} / det(g)
det_g = M_numeric_sym.det()
# Note: indices are 0-based in Python (1,2,1,2 becomes 0,1,0,1)
K_sym = RiemannCovar[0, 1, 0, 1] / det_g
K_sym = sp.simplify(K_sym)

print("Gaussian Curvature (K) symbolic expression computed.")
sp.pprint(K_sym.evalf(3))

# 17. Plot K over the range [0, 2*pi]
# Since K only depends on Theta2 in this model:
K_func = sp.lambdify(Theta[1], K_sym.subs(rob_values), "numpy")

theta2_vals = np.linspace(0, 2 * np.pi, 100)
K_vals = K_func(theta2_vals)

plt.figure(figsize=(10, 6))
plt.plot(theta2_vals, K_vals, lw=2, color='blue')
plt.title("Gaussian Curvature $K$ of Robot C-Space vs. $\Theta_2$")
plt.xlabel("$\Theta_2$ (Joint 2 Angle in Radians)")
plt.ylabel("Curvature $K$")
plt.grid(True, linestyle='--')
plt.axhline(0, color='black', lw=1)
plt.show()