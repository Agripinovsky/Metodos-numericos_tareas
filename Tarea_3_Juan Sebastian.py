import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
#EEJERCICIO 1

#Temperatura
xData = np.array([0, 21.1, 37.8, 54.4, 71.1, 87.8, 100])

#Viscocidad
yData = np.array([0.101, 1.79, 1.13, 0.696, 0.519, 0.338, 0.296])

#Viscocidad evaluada en T
interpx = np.array([10, 30, 60, 90])

# Interpolación de newton
def evalPoly(a, xData, x):  
    n = len(xData) - 1  
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - xData[n - k]) * p
    return p

def coeffts(xData, yData):
    m = len(xData)  
    a = yData.copy() 
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1]) / (xData[k:m] - xData[k - 1])  
    return a

cffs = coeffts(xData, yData)

interpy = [evalPoly(cffs, xData, x) for x in interpx]

# Resultados de la interpolación
print("xInterpolado - yInterpolado")
print("--------------------------------")
for i in range(len(interpx)):
    print(" %.1f  %.3f" % (interpx[i], interpy[i]))

#Grafica
plt.scatter(xData, yData, label="Puntos originales", color="blue")
# Interpolación de Newton
plt.plot(interpx, interpy, label="Interpolación de Newton", linestyle="-")
# Graficar los puntos de interpolación
plt.scatter(interpx, interpy, label="Puntos interpolado", color="red", zorder=5)
plt.xlabel("T(°C)")
plt.ylabel("μk(10^-3 m^2/s)")
plt.title("Metodo de interpolación de Newton")
plt.legend()
plt.grid(True)

#EJERCICIO 2

x_points = np.array([0, 1.525, 3.050, 4.575, 6.10, 7.625, 9.150])  # Temperaturas (T en °C)
y_points = np.array([1, 0.8617, 0.7385, 0.6292, 0.5328, 0.4481, 0.3741])  # Viscosidad (μ_k)
xp = 10.5

def lagrange_1(x_points, y_points, xp):
    m = len(x_points)
    n = m - 1
    x = sp.symbols("x")


    def lagrange_basis(xp, x_points, i):
        L_i = 1
        for j in range(len(x_points)):
            if j != i:
                L_i *= (xp - x_points[j]) / (x_points[i] - x_points[j])
        return L_i
#Polinomio de Lagrange
    def lagrange_interpolation(xp, x_points, y_points):
        yp = 0
        for i in range(len(x_points)):
            yp += y_points[i] * lagrange_basis(xp, x_points, i)
        return yp

    # Calcular el valor interpolado
    yp = lagrange_interpolation(xp, x_points, y_points)
    print("For x = %.1f, y = %.1f" % (xp, yp))

    x_interpol = np.linspace(min(x_points), max(x_points), 100)
    y_interpol = [
        lagrange_interpolation(x_val, x_points, y_points) for x_val in x_interpol
    ]

#Grafica
    plt.scatter(x_points, y_points, label="Puntos Originales", color="blue")
    plt.plot(
        x_interpol, y_interpol, label="Interpolación de Lagrange", linestyle="-"
    )

    plt.scatter(xp, yp, label="Punto interpolado", color="red", zorder=5)
    plt.text(xp, yp, f"({xp:.1f}, {yp:.1f})", fontsize=12, verticalalignment="bottom")

    # Añadir etiquetas y leyenda
    plt.xlabel("h(km)")
    plt.ylabel("Densidad Relativa")
    plt.title("Polinomio de Interpolación de Lagrange")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Construccion del polinomio
    polinomio = 0
    for i in range(len(x_points)):
        z = y_points[i]
        for j in range(len(x_points)):
            if j != i:
                z *= (x - x_points[j]) / (x_points[i] - x_points[j])
        polinomio += z

    pol = sp.simplify(polinomio)
    print("Polinomio de Interpolación de Lagrange:")
    print(f"y(x) = {polinomio}")
    print("\nPolinomio:")
    print(f"y(x) = {pol}")
    return yp
lagrange_1(x_points, y_points, xp)

#EJERCICIO 3 

def evalPoly(a, xData, x): 
    n = len(xData) - 1  
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - xData[n - k]) * p
    return p


def coeffts(xData, yData):
    m = len(xData)  # Número de datos
    a = yData.copy()
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1]) / (xData[k:m] - xData[k - 1])
    return a

xData = np.array([0,400,800,1200,1600])
yData = np.array([0,0.072,0.233,0.712,3.400])
coeff = coeffts(xData, yData)
x = np.arange(0, 2550, 100)
plt.plot(x, evalPoly(coeff, xData, x), "b", label="Newton")
plt.plot(xData, yData, "o", label="Datos", color="black")
plt.xlabel('V (rpm)')
plt.ylabel('A (mm)')
plt.legend()
plt.grid()
plt.show()

print("  x    yExacta        yInt       Error(%)")
print("------------------------------------------")
for i in range(len(x)):
    y = evalPoly(coeff, xData, x[i])
    yExacta = 4.8 * (math.cos(math.pi * x[i]) / 20)
    Error = abs(((yExacta - y) / yExacta) * 100)
    print(" %.1f  %.8f   %.8f    %.8f" % (x[i], yExacta, y, Error))

