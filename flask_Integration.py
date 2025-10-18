
import numpy as np
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    # Get input parameters from form
    L = float(request.form['length'])
    Nt = int(float(request.form['time-step']))  # Convert to int for time steps
    Nx = int(float(request.form['spatial-steps']))  # Convert to int for spatial points
    alpha = float(request.form['alpha'])
    M=int(request.form['no_of_steps'])  # Consider making this configurable too
    t_max= float(request.form['t_max'])  # Maximum time for simulation
    # Spatial grid
    x = np.linspace(0, L, Nx)
    
    # Time parameters - consider making t_max configurable too
    
    t_values = np.linspace(0, t_max, Nt)
    
    # Initial condition
    def initial_condition(x, L):
        return np.where(x < L/2, 2*x/L, 2*(L-x)/L)  #if true select 1st condition , if false select 2nd condition
    u0 = initial_condition(x, L)
    
    # Compute Fourier coefficients
    def compute_fourier_coeffs(u0, L, M):
        B = np.zeros(M)
        dx = L / (Nx - 1)
        for n in range(1, M+1):
            integrand = u0 * np.sin(n * np.pi * x / L)
            B[n-1] = (2/L) * np.trapz(integrand, dx=dx)
        return B
    B_n = compute_fourier_coeffs(u0, L, M)
    
    # Fourier solution at any time t
    def fourier_solution(x, t, B_n, L, alpha, M):
        u = np.zeros_like(x)
        for n in range(1, M+1):
            k = n * np.pi / L
            u += B_n[n-1] * np.sin(k * x) * np.exp(-alpha * k**2 * t)
        return u
    
    # Compute solution history at all time steps
    u_history = [fourier_solution(x, t, B_n, L, alpha, M) for t in t_values]
    
    # Return data as JSON
    return jsonify({
        'x': x.tolist(),
        't_values': t_values.tolist(),  # Include time points
        'u_history': [u.tolist() for u in u_history],  # Full time evolution
        'u_final': u_history[-1].tolist(),  # Final state (t = t_max)
        'B_n': B_n.tolist()
    })


@app.route('/calculate2d', methods=['POST'])
def calculate2d():
    Lx = float(request.form['Lx'])
    Ly = float(request.form['Ly'])
    alpha = float(request.form['alpha2d'])
    Nx = int(request.form['Nx'])
    Ny = int(request.form['Ny'])
    Nt = int(request.form['Nt'])
    M = int(request.form['M2d'])
    t_max = float(request.form['tmax2d'])

    # Spatial grid
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Time grid
    time = np.linspace(0, t_max, Nt)

    # Initial condition
    def initial_condition(x, y):
        return np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly)
    u0 = initial_condition(X, Y)

    # Compute Fourier coefficients
    def compute_fourier_coeffs(u0, Lx, Ly, M):
        A = np.zeros((M, M))
        dx = Lx / (Nx-1)
        dy = Ly / (Ny-1)
        for m in range(1, M+1):
            for n in range(1, M+1):
                integrand = u0 * np.sin(m*np.pi*X/Lx) * np.sin(n*np.pi*Y/Ly)
                A[m-1, n-1] = (4/(Lx*Ly)) * np.trapz(np.trapz(integrand, dx=dy, axis=0), dx=dx)
        return A
    A_mn = compute_fourier_coeffs(u0, Lx, Ly, M)

    # Solution function
    def fourier_solution(x, y, t, A_mn, Lx, Ly, alpha, M):
        u = np.zeros_like(x)
        for m in range(1, M+1):
            for n in range(1, M+1):
                kx = m * np.pi / Lx
                ky = n * np.pi / Ly
                lambda_mn = kx**2 + ky**2
                u += A_mn[m-1, n-1] * np.sin(kx*x) * np.sin(ky*y) * np.exp(-alpha * lambda_mn * t)
        return u

    # Compute solution at each time step for animation
    u_history = []
    for t in time:
        U = fourier_solution(X, Y, t, A_mn, Lx, Ly, alpha, M)
        u_history.append(U.tolist())

    # Return data for animation
    # ...existing code...
    return jsonify({
        'x': x.tolist(),
        'y': y.tolist(),
        'u_history': u_history,
        't_values': time.tolist(),
        'A_mn': A_mn.tolist()  # <-- Add this line
    })

if __name__ == '__main__':
    app.run(debug=True)