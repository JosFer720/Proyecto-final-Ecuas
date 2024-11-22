"""

                                                     Documentación Interna - PROYECTO 02

    Nombre del Programa: Proyecto.py

    Fin En Mente: Resolver al menos una ED de primer orden, una ED de segundo orden y un sistema de EDs de 2x2 con el método de Euler Generalizado.

    Programador: Diego Fabián Morales Dávila    | 23267
                 Erick Antonio Guerra Illescas  | 23208
                 José Fernando Ruiz Estrada     | 23065

    Lenguaje: Python 3.11

    Recursos: Ninguno

"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Método de Euler Generalizado
def euler(f, y0, t0, tf, h):
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = y[i-1] + h * np.array(f(t[i-1], y[i-1]))
    return t, y

# Resolver una ED de Primer Orden
def resolver_primer_orden():
    print("\nResolviendo una EDO de Primer Orden:")
    expr = input("Introduce la ecuación diferencial (por ejemplo, -2*y + sin(t)): ")
    y, t = sp.symbols('y t')
    f_expr = sp.sympify(expr)  
    f = sp.lambdify([t, y], f_expr, "numpy")  

    y0 = float(input("Introduce la condición inicial y(0): "))
    t0 = float(input("Introduce el tiempo inicial t0: "))
    tf = float(input("Introduce el tiempo final tf: "))
    h = float(input("Introduce el paso de tiempo h: "))

    # Función para Método de Euler
    def edo_primer_orden(t, y):
        return [f(t, y[0])]

    t_vals, y_vals = euler(edo_primer_orden, [y0], t0, tf, h)

    print("\nSolución (y):")
    print(y_vals[:, 0])

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, y_vals[:, 0], label="Aproximación por Euler", marker="o")
    # Graficar la solución exacta si es posible
    try:
        exact_sol = sp.lambdify(t, sp.dsolve(sp.Eq(y.diff(t), f_expr), y).rhs, "numpy")
        plt.plot(t_vals, exact_sol(t_vals), label="Solución Exacta", linestyle="--")
    except:
        print("No se pudo calcular la solución exacta automáticamente.")
    plt.title("Resolución de EDO de Primer Orden")
    plt.xlabel("Tiempo t")
    plt.ylabel("Solución y(t)")
    plt.legend()
    plt.grid()
    plt.show()

# Resolver una ED de Segundo Orden
def resolver_segundo_orden():
    print("\nResolviendo una EDO de Segundo Orden:")
    expr = input("Introduce la ecuación diferencial en términos de y y d2y/dt2 (por ejemplo, -y - dy): ")
    y, dy, t = sp.symbols('y dy t')
    f_expr = sp.sympify(expr)  
    f = sp.lambdify([t, y, dy], f_expr, "numpy")  

    y0 = float(input("Introduce la condición inicial y(0): "))
    dy0 = float(input("Introduce la condición inicial dy/dt(0): "))
    t0 = float(input("Introduce el tiempo inicial t0: "))
    tf = float(input("Introduce el tiempo final tf: "))
    h = float(input("Introduce el paso de tiempo h: "))

    # Función para Método de Euler (convertir a sistema de primer orden)
    def edo_segundo_orden(t, z):
        y, dy = z
        d2y = f(t, y, dy)
        return [dy, d2y]

    t_vals, y_vals = euler(edo_segundo_orden, [y0, dy0], t0, tf, h)

    print("\nSolución (y):")
    print(y_vals[:, 0])

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, y_vals[:, 0], label="Aproximación por Euler", marker="o")
    plt.title("Resolución de EDO de Segundo Orden")
    plt.xlabel("Tiempo t")
    plt.ylabel("Solución y(t)")
    plt.legend()
    plt.grid()
    plt.show()

# Resolver un Sistema de EDs de 2x2
def resolver_sistema_2x2():
    print("\nResolviendo un Sistema de EDs de 2x2:")
    expr1 = input("Introduce la ecuación dx/dt (por ejemplo, 3*x + 5*y): ")
    expr2 = input("Introduce la ecuación dy/dt (por ejemplo, -2*x + 4*y): ")
    x, y, t = sp.symbols('x y t')
    f1_expr = sp.sympify(expr1)
    f2_expr = sp.sympify(expr2)
    f1 = sp.lambdify([t, x, y], f1_expr, "numpy")
    f2 = sp.lambdify([t, x, y], f2_expr, "numpy")

    x0 = float(input("Introduce la condición inicial x(0): "))
    y0 = float(input("Introduce la condición inicial y(0): "))
    t0 = float(input("Introduce el tiempo inicial t0: "))
    tf = float(input("Introduce el tiempo final tf: "))
    h = float(input("Introduce el paso de tiempo h: "))

    # Función para Método de Euler
    def sistema_2x2(t, z):
        x, y = z
        dxdt = f1(t, x, y)
        dydt = f2(t, x, y)
        return [dxdt, dydt]

    t_vals, z_vals = euler(sistema_2x2, [x0, y0], t0, tf, h)

    print("\nSolución (x, y):")
    print("x(t):", z_vals[:, 0])
    print("y(t):", z_vals[:, 1])

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, z_vals[:, 0], label="x(t) - Aproximación por Euler", marker="o")
    plt.plot(t_vals, z_vals[:, 1], label="y(t) - Aproximación por Euler", marker="s")
    plt.title("Sistema de EDs de 2x2")
    plt.xlabel("Tiempo t")
    plt.ylabel("Solución")
    plt.legend()
    plt.grid()
    plt.show()

# Menú Principal
def menu():
    print("Elige el tipo de ecuación a resolver:")
    print("1. Ecuación Diferencial de Primer Orden")
    print("2. Ecuación Diferencial de Segundo Orden")
    print("3. Sistema de EDs de 2x2")

    opcion = int(input("Introduce tu opción (1, 2 o 3): "))
    if opcion == 1:
        resolver_primer_orden()
    elif opcion == 2:
        resolver_segundo_orden()
    elif opcion == 3:
        resolver_sistema_2x2()
    else:
        print("Opción inválida.")

# Ejecutar el menú
menu()