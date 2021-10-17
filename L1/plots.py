import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from gradient import gradient_descent, f, ALPHA, EPSILON, funct_der, MAX_IT, X0, funct


fig, axis = plt.subplots(2, 2)

for ax, alpha, x0 in zip(axis.reshape(4), [0.01, 0.01, 0.05, 0.07], [-2.5, 3, 3, -2.5]):
    steps = gradient_descent(funct_der, x0, MAX_IT, alpha, EPSILON)
    x_vals = np.linspace(-4, 4)
    y_vals = f(x_vals)
    ax.plot(x_vals, y_vals, label="$x^4-5x^2-3x$\n")
    ax.plot(steps, f(np.array(steps)), "--.")
    ax.plot(steps[-1], f(np.array(steps[-1])), "ro",
             label=f"$({round(steps[-1], 5)},{round(f(np.array(steps[-1])), 5)})$")
    ax.set_title(fr"$\alpha={alpha}, x_0={x0}$")
    ax.set(xlim=(-4, 4), ylim=(-15, 30))
    ax.legend()
plt.show()
