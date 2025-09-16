from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt


# Definimos la condición de Armijo
def condicion_armijo(f, x, p, alpha, c1):
    return f(x + alpha * p) <= f(x) + c1 * alpha * grad(f)(x).dot(p)


# Definimos la condición de Wolfe
def condicion_wolfe(f, x, p, alpha, c2):
    return grad(f)(x + alpha * p).dot(p) >= c2 * grad(f)(x).dot(p)


# Algoritmo de búsqueda de paso
def busqueda_paso(f, x, p, alpha, c1, c2, rho):
    # Verificamos que los valores de los parámetros sean válidos
    if c1 <= 0 or c1 >= 1:
        print('El valor de c1 no es válido')
        return
    if c2 <= c1 or c2 >= 1:
        print('El valor de c2 no es válido')
        return
    if rho <= 0 or rho >= 1:
        print('El valor de rho no es válido')
        return
    # Buscamos el tamaño de paso
    while not (condicion_wolfe(f, x, p, alpha, c2) and condicion_armijo(f, x, p, alpha, c1)):
        alpha = rho * alpha
    return alpha


# EJEMPLO 1
def f1(x):
    return np.cos(x + np.pi / 4)


# Llamamos a la función de búsqueda de paso
initial_x = np.array([0.0])
p = np.array([1.0])
alpha = 5.0
c1 = 10 ** -4
c2 = 0.9
rho = 0.75
alpha = busqueda_paso(f1, initial_x, p, alpha, c1, c2, rho)
print('Ejemplo 1: f(x) = cos(x + pi/4) con rho = 0.75, alpha = 5.0')
print('El tamaño de paso es:', alpha)

# Graficamos la función, el punto inicial y el tamaño de paso
x = np.linspace(-np.pi, 3 * np.pi, 100)
y = np.cos(x + np.pi / 4)
plt.plot(x, y)
plt.scatter(initial_x, f1(initial_x), color='r')
plt.plot([initial_x, initial_x + alpha * p], [f1(initial_x), f1(initial_x + alpha * p)], color='g')
plt.legend(['f(x)', 'Punto inicial', 'Tamaño de paso'])
plt.title('Ejemplo 1: f(x) = cos(x + pi/4) con rho = 0.75, alpha = 5.0')
plt.show()

# EJEMPLO 2: Cambio de valor rho
alpha = 5.0
rho = 0.5
alpha = busqueda_paso(f1, initial_x, p, alpha, c1, c2, rho)
print('Ejemplo 2: f(x) = cos(x + pi/4) con rho = 0.5, alpha = 5.0')
print('El tamaño de paso es:', alpha)

# Graficamos la función, el punto inicial y el tamaño de paso
x = np.linspace(-np.pi, 3 * np.pi, 100)
y = np.cos(x + np.pi / 4)
plt.plot(x, y)
plt.scatter(initial_x, f1(initial_x), color='r')
plt.plot([initial_x, initial_x + alpha * p], [f1(initial_x), f1(initial_x + alpha * p)], color='g')
plt.legend(['f(x)', 'Punto inicial', 'Tamaño de paso'])
plt.title('Ejemplo 2: f(x) = cos(x + pi/4) con rho = 0.5, alpha = 5.0')
plt.show()

# EJEMPLO 3: Cambio de valor alpha y rho
alpha = 2.0
rho = 0.5
alpha = busqueda_paso(f1, initial_x, p, alpha, c1, c2, rho)
print('Ejemplo 3: f(x) = cos(x + pi/4) con rho = 0.5, alpha = 2.0')
print('El tamaño de paso es:', alpha)

# Graficamos la función, el punto inicial y el tamaño de paso
x = np.linspace(-np.pi, 3 * np.pi, 100)
y = np.cos(x + np.pi / 4)
plt.plot(x, y)
plt.scatter(initial_x, f1(initial_x), color='r')
plt.plot([initial_x, initial_x + alpha * p], [f1(initial_x), f1(initial_x + alpha * p)], color='g')
plt.legend(['f(x)', 'Punto inicial', 'Tamaño de paso'])
plt.title('Ejemplo 3: f(x) = cos(x + pi/4) con rho = 0.5, alpha = 2.0')
plt.show()

# EJEMPLO 4: Cambio de valor alpha
alpha = 2.0
rho = 0.75
alpha = busqueda_paso(f1, initial_x, p, alpha, c1, c2, rho)
print('Ejemplo 4: f(x) = cos(x + pi/4) con rho = 0.75, alpha = 2.0')
print('El tamaño de paso es:', alpha)

# Graficamos la función, el punto inicial y el tamaño de paso
x = np.linspace(-np.pi, 3 * np.pi, 100)
y = np.cos(x + np.pi / 4)
plt.plot(x, y)
plt.scatter(initial_x, f1(initial_x), color='r')
plt.plot([initial_x, initial_x + alpha * p], [f1(initial_x), f1(initial_x + alpha * p)], color='g')
plt.legend(['f(x)', 'Punto inicial', 'Tamaño de paso'])
plt.title('Ejemplo 4: f(x) = cos(x + pi/4) con rho = 0.75, alpha = 2.0')
plt.show()


# EJEMPLO 5
def f2(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0] ** 2)


# Llamamos a la función de búsqueda de paso
initial_x = np.array([2.0, 3.0])
p = np.array([-1.0, -1.0])
alpha = 1.0
c1 = 10 ** -4
c2 = 0.9
rho = 0.75
alpha = busqueda_paso(f2, initial_x, p, alpha, c1, c2, rho)
print('Ejemplo 5: f(x) = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0] ** 2) con rho = 0.75, alpha = 1.0')
print('El tamaño de paso es:', alpha)

# Graficamos la función, el punto inicial y el tamaño de paso
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = 100 * (Y - X ** 2) ** 2 + (1 - X ** 2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.scatter(initial_x[0], initial_x[1], f2(initial_x), color='r')
ax.quiver(initial_x[0], initial_x[1], f2(initial_x), p[0], p[1], 0, color='r')
ax.scatter(initial_x[0] + alpha * p[0], initial_x[1] + alpha * p[1], f2(initial_x + alpha * p), color='g')
plt.title('f(x) = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0] ** 2)')
plt.show()

# Grafico el paso en la curva de nivel
x = np.linspace(-5, 5, 100)
y = np.linspace(-10, 5, 100)
X, Y = np.meshgrid(x, y)
Z = 100 * (Y - X ** 2) ** 2 + (1 - X ** 2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contour(X, Y, Z, 20)
ax.scatter(initial_x[0], initial_x[1], color='r')
ax.plot([initial_x[0], initial_x[0] + alpha * p[0]], [initial_x[1], initial_x[1] + alpha * p[1]], color='g')
plt.legend(['Punto inicial', 'Tamaño de paso'])
plt.title('Ejemplo 5: f(x) = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0] ** 2)')
plt.suptitle('con rho = 0.75, alpha = 1.0')
plt.show()

# EJEMPLO 6: Cambio de valor rho
alpha = 1.0
rho = 0.5
alpha = busqueda_paso(f2, initial_x, p, alpha, c1, c2, rho)
print('Ejemplo 6: f(x) = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0] ** 2) con rho = 0.5, alpha = 1.0')
print('El tamaño de paso es:', alpha)

# Grafico el paso en la curva de nivel
x = np.linspace(-5, 5, 100)
y = np.linspace(-10, 5, 100)
X, Y = np.meshgrid(x, y)
Z = 100 * (Y - X ** 2) ** 2 + (1 - X ** 2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contour(X, Y, Z, 20)
ax.scatter(initial_x[0], initial_x[1], color='r')
ax.plot([initial_x[0], initial_x[0] + alpha * p[0]], [initial_x[1], initial_x[1] + alpha * p[1]], color='g')
plt.legend(['Punto inicial', 'Tamaño de paso'])
plt.title('Ejemplo 6: f(x) = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0] ** 2)')
plt.suptitle('con rho = 0.5, alpha = 1.0')
plt.show()

# EJEMPLO 7: Cambio de valor alpha
alpha = 3.0
rho = 0.75
alpha = busqueda_paso(f2, initial_x, p, alpha, c1, c2, rho)
print('Ejemplo 7: f(x) = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0] ** 2) con rho = 0.75, alpha = 3.0')
print('El tamaño de paso es:', alpha)

# Grafico el paso en la curva de nivel
x = np.linspace(-5, 5, 100)
y = np.linspace(-10, 5, 100)
X, Y = np.meshgrid(x, y)
Z = 100 * (Y - X ** 2) ** 2 + (1 - X ** 2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contour(X, Y, Z, 20)
ax.scatter(initial_x[0], initial_x[1], color='r')
ax.plot([initial_x[0], initial_x[0] + alpha * p[0]], [initial_x[1], initial_x[1] + alpha * p[1]], color='g')
plt.legend(['Punto inicial', 'Tamaño de paso'])
plt.title('Ejemplo 7: f(x) = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0] ** 2)')
plt.suptitle('con rho = 0.75, alpha = 3.0')
plt.show()

# EJEMPLO 8: Cambio de valor alpha y rho
alpha = 3.0
rho = 0.5
alpha = busqueda_paso(f2, initial_x, p, alpha, c1, c2, rho)
print('Ejemplo 8: f(x) = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0] ** 2) con rho = 0.5, alpha = 3.0')
print('El tamaño de paso es:', alpha)

# Grafico el paso en la curva de nivel
x = np.linspace(-5, 5, 100)
y = np.linspace(-10, 5, 100)
X, Y = np.meshgrid(x, y)
Z = 100 * (Y - X ** 2) ** 2 + (1 - X ** 2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contour(X, Y, Z, 20)
ax.scatter(initial_x[0], initial_x[1], color='r')
ax.plot([initial_x[0], initial_x[0] + alpha * p[0]], [initial_x[1], initial_x[1] + alpha * p[1]], color='g')
plt.legend(['Punto inicial', 'Tamaño de paso'])
plt.title('Ejemplo 8: f(x) = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0] ** 2)')
plt.suptitle('con rho = 0.5, alpha = 3.0')
plt.show()
