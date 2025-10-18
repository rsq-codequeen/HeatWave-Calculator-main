import numpy as np
import matplotlib.pyplot as plt

L = 1.0
x = np.linspace(0, L, 20)
u0 = np.where(x < L/2, 2*x/L, 2*(L-x)/L)

plt.plot(x, u0, 'r-', linewidth=2)
plt.title("Initial Temperature Profile")
plt.xlabel("Position (x)"); plt.ylabel("Temperature (u)")
plt.grid()
plt.show()