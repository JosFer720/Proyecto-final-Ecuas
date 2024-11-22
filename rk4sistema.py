import numpy as np
import matplotlib.pyplot as plt

def procesar_sistema(ecuaciones_input):
    ecuaciones = ecuaciones_input.split(',')
    num_ecuaciones = len(ecuaciones)
    def sistema(t, variables):
        if len(variables) != num_ecuaciones:
            raise ValueError(f"El número de ecuaciones ({num_ecuaciones}) no coincide con las condiciones iniciales ({len(variables)})")
        x = variables
        return np.array([eval(eq) for eq in ecuaciones])
    return sistema

def rk4_system(f, y0, t0, tf, h):
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = h * f(t[i - 1], y[i - 1])
        k2 = h * f(t[i - 1] + h / 2, y[i - 1] + k1 / 2)
        k3 = h * f(t[i - 1] + h / 2, y[i - 1] + k2 / 2)
        k4 = h * f(t[i - 1] + h, y[i - 1] + k3)
        y[i] = y[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, y

def main():
    print("\n--- Sistema de EDOs ---")
    ecuaciones_input = input("Ingresa las ecuaciones separadas por comas (usa 'x[0]', 'x[1]', etc. para variables y 't' para el tiempo, por ejemplo: '3*x[0] + 5*x[1] + np.sin(t), -2*x[0] + 4*x[1] + np.cos(t)': ")
    f = procesar_sistema(ecuaciones_input)
    y0 = list(map(float, input("Ingresa las condiciones iniciales para las variables (por ejemplo, '1, 0'): ").split(',')))
    t0 = float(input("Ingresa el valor inicial de t (t0): "))
    tf = float(input("Ingresa el valor final de t (tf): "))
    h = float(input("Ingresa el tamaño del paso (h): "))

    t, y = rk4_system(f, y0, t0, tf, h)

    print("\nValores de t y las variables:")
    print(f"{'t':<10}" + "".join([f"x[{i}]:<15" for i in range(y.shape[1])]))
    for i in range(len(t)):
        print(f"{t[i]:<10.4f}" + "".join([f"{y[i, j]:<15.4f}" for j in range(y.shape[1])]))

    for i in range(y.shape[1]):
        plt.plot(t, y[:, i], label=f"x[{i}]")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Variables")
    plt.title("Solución del sistema de EDOs")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
