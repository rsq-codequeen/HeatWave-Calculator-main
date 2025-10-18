import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
L = 2.0          # Length of the rod
alpha = 0    # Thermal diffusivity : higher value , faster diffusion
Nx = 100         # Number of spatial points
Nt = 200         # Number of time steps
M = 5           # Number of Fourier modes to use

# Spatial and time grids
x = np.linspace(0, L, Nx)
t_max = 1.0
time = np.linspace(0, t_max, Nt)


# Initial condition
def initial_condition(x, L):
    return np.where(x < L/2, 2*x/L, 2*(L-x)/L)


u0 = initial_condition(x, L)

# Compute Fourier coefficients
#Bn= L2∫ 0L u(x,0)sin( Lnπx )dx
def compute_fourier_coeffs(u0, L, M):
    B = np.zeros(M)
    dx = L / (Nx - 1) 

    for n in range(1, M+1):
        integrand = u0 * np.sin(n * np.pi * x / L) #u0 = f(x)
        B[n-1] = (2/L) * np.trapz(integrand, dx=dx)
    return B

B_n = compute_fourier_coeffs(u0, L, M)

print("Fourier coefficient Bn:", B_n)
# Fourier series solution
def fourier_solution(x, t, B_n, L, alpha, M):
    u = np.zeros_like(x)
    for n in range(1, M+1):
        k = n * np.pi / L
        u += B_n[n-1] * np.sin(k * x) * np.exp(-alpha * k**2 * t)
    return u

# Animate
fig, ax = plt.subplots()
line, = ax.plot(x, u0)

def update(frame):
    t = frame * t_max / (Nt-1)
    u = fourier_solution(x, t, B_n, L, alpha, M)
    line.set_ydata(u)
    ax.set_title(f"1D Heat Equation at t = {t:.2f}")
    ax.set_ylim(0, 1)
    return line,

ani = FuncAnimation(fig, update, frames=Nt, interval=50, blit=True)
plt.show()