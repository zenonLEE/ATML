import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def Gcal_AT(P, T):
    P = np.array(P).reshape(-1)
    T = np.array(T).reshape(-1)

    qms = np.array([6.0492, 4.71918, 3.55639])
    Ks = np.array([5.00526, 2.6474, 1.88271])
    Psat = np.array([34.713, 64.121, 83.320])
    R = 8.314 / 1000

    delta_G_matrix = np.zeros((len(qms), len(P)))
    q_matrix = np.zeros((len(qms), len(P)))

    for i in range(len(qms)):
        q = qms[i] * Ks[i] * P / (1 + Ks[i] * P)
        delta_G = q * R * T[i] * np.log(Psat[i] / P)
        delta_G_matrix[i, :] = -delta_G
        q_matrix[i, :] = q

    return delta_G_matrix, q_matrix


def Gcal_EL(P, T):
    P = np.array(P).reshape(-1)
    T = np.array(T).reshape(-1)

    qms = np.array([6.0492, 4.71918, 3.55639])
    Ks = np.array([5.00526, 2.6474, 1.88271])
    R = 8.314 / 1000

    delta_G_matrix = np.zeros((len(qms), len(P)))
    q_matrix = np.zeros((len(qms), len(P)))

    for i in range(len(qms)):
        q = qms[i] * Ks[i] * P / (1 + Ks[i] * P)
        delta_G = -q * R * T[i] * np.log(Ks[i])
        delta_G_matrix[i, :] = delta_G
        q_matrix[i, :] = q

    return delta_G_matrix, q_matrix


def Gcal_M(P, T):
    P = np.array(P).reshape(-1)
    T = np.array(T).reshape(-1)

    qms = np.array([6.0492, 4.71918, 3.55639])
    Ks = np.array([5.00526, 2.6474, 1.88271])
    R = 8.314 / 1000

    delta_G_matrix1 = np.zeros((len(qms), len(P)))
    q_matrix = np.zeros((len(qms), len(P)))

    for i in range(len(qms)):
        q = qms[i] * Ks[i] * P / (1 + Ks[i] * P)
        result = np.zeros(len(P))

        for j in range(len(P)):
            lnP = lambda a: qms[i] * Ks[i] / (1 + Ks[i] * a)
            result[j], _ = quad(lnP, 0, P[j])

        q_matrix[i, :] = q
        delta_G = R * T[i] * result
        delta_G_matrix1[i, :] = -delta_G

    delta_G_matrix = delta_G_matrix1 / q_matrix
    return delta_G_matrix, q_matrix, delta_G_matrix1


def fitting(P, T, delta_G_matrix):
    slopes = np.zeros(delta_G_matrix.shape[1])

    for i in range(delta_G_matrix.shape[1]):
        delta_G_column = delta_G_matrix[:, i]
        coefficients = np.polyfit(T, delta_G_column, 1)
        slopes[i] = coefficients[0]

    delta_S = -slopes
    return delta_S


def plot_q_vs_P(P, T, q_matrix):
    colors = ['b', 'g', 'r']
    plt.figure()
    for i in range(len(T)):
        plt.plot(P, q_matrix[i, :], color=colors[i], linewidth=2, label=f'T={T[i]}K')
    plt.xlabel('P')
    plt.ylabel('q')
    plt.legend()
    plt.show()


# Example usage:
P = np.arange(0.01, 1.01, 0.01)
T = np.array([273, 298, 323])

delta_G_matrix, q_matrix = Gcal_AT(P, T)
delta_S = fitting(P, T, delta_G_matrix)
plot_q_vs_P(P, T, q_matrix)
