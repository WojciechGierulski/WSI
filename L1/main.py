from sympy import Symbol, lambdify
import matplotlib.pyplot as plt


X0 = 0
MAX_IT = 1000
ALPHA = 0.01
EPSILON = 0.001

x = Symbol("x")
# Function
f = x**2 + 3 * x + 8
funct_der = f.diff(x)
f = lambdify(x, f)
funct_der = lambdify(x, funct_der)


def gradient_descent(funct_der, x0, max_it, alpha, epsilon):
    i=0
    current_x = x0
    next_x = None
    while True:
        next_x = current_x - alpha * funct_der(current_x)
        # stop test
        if abs(funct_der(current_x)) <= epsilon or i >= max_it:
            return current_x
        i += 1
        current_x = next_x



a = gradient_descent(funct_der, X0, MAX_IT, ALPHA, EPSILON)
print(a)



