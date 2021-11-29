from sympy import Symbol, lambdify

X0 = 0
MAX_IT = 1000
ALPHA = 0.02
EPSILON = 0.01

x = Symbol("x")

# Function
funct = x ** 4 - 5 * (x ** 2) - 3 * x
funct_der = funct.diff(x)
f = lambdify(x, funct, modules=['numpy'])
funct_der = lambdify(x, funct_der)


def gradient_descent(funct_der, x0, max_it, alpha, epsilon):
    i = 0
    current_x = x0
    next_x = None
    x_list = []
    x_list.append(current_x)
    while True:
        next_x = current_x - alpha * funct_der(current_x)
        x_list.append(next_x)
        # stop test
        if abs(funct_der(current_x)) <= epsilon or i >= max_it:
            return x_list
        i += 1
        current_x = next_x

if __name__ == "__main__":
    pts = gradient_descent(funct_der, X0, MAX_IT, ALPHA, EPSILON)
    print(pts[-1])