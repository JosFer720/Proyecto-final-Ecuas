import numpy as np
import matplotlib.pyplot as plt
import math

def procesar_ecuacion(ecuacion_str, variables):
    """
    Convierte una ecuación en una función lambda con las variables especificadas.

    Args:
        ecuacion_str: Cadena de texto de la ecuación.
        variables: Lista de variables esperadas (por ejemplo, ["t", "x", "y"]).

    Returns:
        Una función lambda evaluable con las variables dadas.
    """
    contexto = {"math": math, "log": math.log, "sin": math.sin, "cos": math.cos, 
                "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan, 
                "e": math.e}
    contexto.update({var: None for var in variables})  # Asegurar las variables en el contexto

    try:
        # Convertir la ecuación a una función lambda
        funcion = eval(f"lambda {', '.join(variables)}: {ecuacion_str}", contexto)
        return funcion
    except Exception as e:
        raise ValueError(f"Error al procesar la ecuación: {e}")

def procesar_sistema(ecuaciones_str):
    """
    Convierte una lista de ecuaciones separadas por comas en funciones lambda.

    Args:
        ecuaciones_str: Cadena con ecuaciones separadas por comas (por ejemplo, "3*x + 5*y + math.sin(t), -2*x + 4*y + math.cos(t)").

    Returns:
        Lista de funciones lambda que representan el sistema.
    """
    try:
        ecuaciones = ecuaciones_str.split(",")
        return [procesar_ecuacion(eq.strip(), ["t", "x", "y"]) for eq in ecuaciones]
    except Exception as e:
        raise ValueError(f"Error al procesar el sistema: {e}")

def rk4_simple(f, y0, x0, xf, h):
    """
    Método RK4 para ecuaciones diferenciales de primer orden.

    Args:
        f: función que describe la ecuación (dy/dx = f(x, y)).
        y0: condición inicial.
        x0: valor inicial de x.
        xf: valor final de x.
        h: tamaño del paso.

    Returns:
        xs: valores de x.
        ys: valores de y.
    """
    xs = np.arange(x0, xf + h, h)
    ys = np.zeros(len(xs))
    ys[0] = y0

    for i in range(len(xs) - 1):
        x = xs[i]
        y = ys[i]

        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)

        ys[i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return xs, ys

def rk4_sistema_generalizado(fns, y0, t0, tf, h):
    """
    Resuelve un sistema de ecuaciones diferenciales con el método RK4.

    Args:
        fns: Lista de funciones que definen el sistema (dy/dt = f(t, y)).
        y0: Vector de condiciones iniciales.
        t0: Tiempo inicial.
        tf: Tiempo final.
        h: Tamaño del paso.

    Returns:
        ts: Array de valores de t.
        ys: Array de soluciones, donde cada columna corresponde a una variable del sistema.
    """
    num_vars = len(fns)
    ts = np.arange(t0, tf + h, h)
    ys = np.zeros((len(ts), num_vars))
    ys[0] = y0

    for i in range(len(ts) - 1):
        t = ts[i]
        y = ys[i]

        k1 = np.array([h * fn(t, *y) for fn in fns])
        k2 = np.array([h * fn(t + h / 2, *(y + k1 / 2)) for fn in fns])
        k3 = np.array([h * fn(t + h / 2, *(y + k2 / 2)) for fn in fns])
        k4 = np.array([h * fn(t + h, *(y + k3)) for fn in fns])

        ys[i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return ts, ys

def main():
    print("\n--- Resolución de Ecuaciones Diferenciales ---")
    print("Selecciona el tipo de ecuación que deseas resolver:")
    print("1. EDO de primer orden")
    print("2. EDO de segundo orden")
    print("3. Sistemas de EDOs")
    opcion = int(input("Ingresa tu opción: "))

    if opcion == 1:
        # EDO de primer orden
        print("\n--- EDO de Primer Orden ---")
        ecuacion_input = input("Ingresa la ecuación en la forma dy/dx = f(x, y): ")
        f = procesar_ecuacion(ecuacion_input, ["x", "y"])
        y0 = float(input("Ingresa la condición inicial y(x0): "))
        x0 = float(input("Ingresa el valor inicial de x (x0): "))
        xf = float(input("Ingresa el valor final de x (xf): "))
        h = float(input("Ingresa el tamaño del paso (h): "))
        xs, ys = rk4_simple(f, y0, x0, xf, h)
        print("\nResultados:")
        for xi, yi in zip(xs, ys):
            print(f"x = {xi:.4f}, y = {yi:.4f}")
        plt.plot(xs, ys, label="y(x)")
        plt.legend()
        plt.title("Solución de la EDO de Primer Orden")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.show()

    elif opcion == 2:
        # EDO de segundo orden
        print("\n--- EDO de Segundo Orden ---")
        ecuacion_input = input("Ingresa la ecuación en la forma d^2y/dx^2 = f(x, y, dy/dx): ")
        f = procesar_ecuacion(ecuacion_input, ["x", "y", "dy_dx"])
        y0 = float(input("Ingresa la condición inicial y(x0): "))
        dy_dx0 = float(input("Ingresa la condición inicial y'(x0): "))
        x0 = float(input("Ingresa el valor inicial de x (x0): "))
        xf = float(input("Ingresa el valor final de x (xf): "))
        h = float(input("Ingresa el tamaño del paso (h): "))
        sistema = [lambda x, y, dy_dx: dy_dx, f]
        xs, ys = rk4_sistema_generalizado(sistema, [y0, dy_dx0], x0, xf, h)
        print("\nResultados:")
        for xi, (yi, dyi_dx) in zip(xs, ys):
            print(f"x = {xi:.4f}, y = {yi:.4f}, y' = {dyi_dx:.4f}")
        plt.plot(xs, ys[:, 0], label="y(x)")
        plt.plot(xs, ys[:, 1], label="y'(x)")
        plt.legend()
        plt.title("Solución de la EDO de Segundo Orden")
        plt.xlabel("x")
        plt.ylabel("Valores")
        plt.grid()
        plt.show()

    elif opcion == 3:
        # Sistema de EDOs
        print("\n--- Sistema de EDOs ---")
        ecuaciones_input = input("Ingresa las ecuaciones separadas por comas (por ejemplo, '3*x + 5*y + sin(t), -2*x + 4*y + cos(t)'): ")
        fns = procesar_sistema(ecuaciones_input)
        y0 = list(map(float, input("Ingresa las condiciones iniciales (por ejemplo, 'x0, y0'): ").split(",")))
        t0 = float(input("Ingresa el valor inicial de t (t0): "))
        tf = float(input("Ingresa el valor final de t (tf): "))
        h = float(input("Ingresa el tamaño del paso (h): "))
        ts, ys = rk4_sistema_generalizado(fns, y0, t0, tf, h)
        print("\nResultados:")
        for ti, valores in zip(ts, ys):
            print(f"t = {ti:.4f}, valores = {valores}")
        for i in range(len(y0)):
            plt.plot(ts, ys[:, i], label=f"Variable {i + 1}")
        plt.legend()
        plt.title("Solución del Sistema de EDOs")
        plt.xlabel("t")
        plt.ylabel("Valores")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()