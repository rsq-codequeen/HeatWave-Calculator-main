import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Parameters
Lx = 1.0  # Length in x-direction
Ly = 1.0  # Length in y-direction
alpha = 0.01  # Thermal diffusivity
Nx = 50   # Number of spatial points in x
Ny = 50   # Number of spatial points in y
Nt = 100  # Number of time steps
M = 20    # Number of Fourier terms to use in each direction

# Spatial grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Time grid
t_max = 1.0
time = np.linspace(0, t_max, Nt)

# Define initial condition (a simple peak in the center)
def initial_condition(x, y):
    return np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly) 
    # For a more general case, you could use:
 

u0 = initial_condition(X, Y)

# Compute Fourier coefficients A_mn
def compute_fourier_coeffs(u0, Lx, Ly, M):
    A = np.zeros((M, M))
    dx = Lx / (Nx-1)
    dy = Ly / (Ny-1)
    
    for m in range(1, M+1):
        for n in range(1, M+1):
            # Numerical integration using trapezoidal rule
            integrand = u0 * np.sin(m*np.pi*X/Lx) * np.sin(n*np.pi*Y/Ly)
            A[m-1, n-1] = (4/(Lx*Ly)) * np.trapz(np.trapz(integrand, dx=dy, axis=0), dx=dx)
    
    return A

A_mn = compute_fourier_coeffs(u0, Lx, Ly, M)

# Solution function using Fourier series
def fourier_solution(x, y, t, A_mn, Lx, Ly, alpha, M):
    u = np.zeros_like(x)
    for m in range(1, M+1):
        for n in range(1, M+1):
            kx = m * np.pi / Lx
            ky = n * np.pi / Ly
            lambda_mn = kx**2 + ky**2
            u += A_mn[m-1, n-1] * np.sin(kx*x) * np.sin(ky*y) * np.exp(-alpha * lambda_mn * t)
    return u

# Create animation
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    t = frame * t_max / (Nt-1)
    U = fourier_solution(X, Y, t, A_mn, Lx, Ly, alpha, M)
    
    surf = ax.plot_surface(X, Y, U, cmap='viridis', rstride=1, cstride=1)
    ax.set_zlim(0, 1)
    ax.set_title(f'2D Heat Equation at t = {t:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature')
    return surf,

ani = FuncAnimation(fig, update, frames=Nt, interval=50, blit=False)
plt.tight_layout()
plt.show()

# To save the animation (requires ffmpeg)
# ani.save('heat_equation_2d.mp4', writer='ffmpeg', fps=10)