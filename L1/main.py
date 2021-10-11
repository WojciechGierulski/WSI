# Pkt początkowy, uczenia

X0 = 0
MAX_IT = 1000
ALPHA = 0.01
EPSILON = 0.001

def funct(x):
    return x**2 + 3 * x + 8

def funct_der(x):
    return 2 * x + 3


def gradient_descent(funct, funct_der, x0, max_it, alpha, epsilon):
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


print("Start")
a = gradient_descent(funct, funct_der, X0, MAX_IT, ALPHA, EPSILON)
print(a)



