# torusControlLaw: Riemannian Geometry & Hamilton-Jacobi-Bellman (HJB) Optimal Control

This repository provides a complete pipeline for analyzing and controlling a 2-link robotic manipulator within its configuration space (c-space), modeled as a 2-torus (T^2).

The project transitions from high-level differential geometry to real-time optimal control using Hamilton-Jacobi-Bellman (HJB) equations.

🚀 Features

    Symbolic Riemannian Metric: Automated derivation of the Mass Matrix (gij​) using Euler-Lagrange equations.

    Curvature Analysis: Computation of Christoffel symbols and the Gaussian Curvature (K) to identify non-Euclidean regions of the C-space.

    HJB Optimal Control: Implementation of an Infinite-Horizon Quadratic Regulator that solves for the optimal torque u∗ on a curved manifold.

🧠 Mathematical Foundation
1. The Metric (gij​)

The robot's kinetic energy defines the Riemannian metric. For this 2-link system, the metric tensor is:
g(θ)=(M11​(θ2​)M21​(θ2​)​M12​(θ2​)M22​​)

2. Gaussian Curvature (K)

We calculate the curvature to understand how "warped" the robot's space is. A non-zero K implies that the shortest path (geodesic) is not a straight line in joint coordinates:
K=det(g)R1212​

3. Optimal Control Law (HJB)

To move from point A to B with minimum energy, we solve the HJB equation for the value function V(q,p):
umin​{L(q,u)+∇VTx˙}=0

The resulting control law u=−R−1BTPx accounts for the varying inertia of the robot as it moves across the torus.

💻 Installation & Usage

    Clone and Setup:
    PowerShell

    git clone https://github.com/DeadbeatJeff/torusControlLaw.git
    cd torusControlLaw
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt

    Run Analysis:

        Execute controLaw.py to simulate the HJB optimal trajectory.
