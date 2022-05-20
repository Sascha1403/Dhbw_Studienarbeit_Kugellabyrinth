import numpy as np


def get_state_space(g_mm, pole):
    g_px = 480/300 * g_mm
    b = 5 / 7 * g_px  # Konstante aus g_mm

    # Matrizen des Zustandsmodells
    A = np.array([[0., 1., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 1.],
                  [0., 0., 0., 0.]])

    B = np.array([[0., 0.],
                  [b, 0.],
                  [0., 0.],
                  [0., b]])

    C = np.array([[1., 0., 0., 0.],
                  [0., 0., 1., 0.]])

    # Berechnung Regelmatrix
    r1 = 7 / (5*g_px) * pole[0] ** 2
    r2 = -2 * 7 / (5*g_px) * pole[1]

    R = np.array([[-r1, 0.],
                  [-r2, 0.],
                  [0., -r1],
                  [0., -r2]]).T

    # Berechnung Vorfilter
    V = np.linalg.inv(np.dot(np.dot(C, np.linalg.inv(np.subtract(np.dot(B, R), A))), B))

    return R, V


def calc_velocity(time, time_last, coords, coords_last):
    vx = (coords[0] - coords_last[0]) / (time - time_last)
    vy = (coords[1] - coords_last[1]) / (time - time_last)

    return (vx, vy)


def servo_y_umrechnung(grad):
    if grad > 10:
        grad = 10
    if grad < -10:
        grad = -10

    t = -0.04343 * grad ** 3 + 0.08153 * grad ** 2 - 32.33 * grad + 1510

    return t


def servo_x_umrechnung(grad):
    if grad > 10:
        grad = 10
    if grad < -10:
        grad = -10

    t = 0.03904 * grad ** 3 + 0.001649 * grad ** 2 + 35.81 * grad + 1496

    return t




l1 = 94.75
l1_5 = 25.75
l2 = 139
l3 = 27.75

p0 = (121.5, l2+l1_5)
p3 = (0, 0)

from matplotlib import pyplot as plt


def servo_umrechnung_neu(alpha_deg):
    """
    if alpha_deg >= 10:
        alpha_deg = 10
    if alpha_deg <= -10:
        alpha_deg = -10
    """
    alpha = alpha_deg * np.pi/180

    p1 = (p0[0] - l1*np.cos(alpha) - l1_5*np.sin(alpha), p0[1] + l1*np.sin(alpha) - l1_5*np.cos(alpha))
    l_hilf = np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
    beta = np.arctan((p1[1] - p3[1]) / (p1[0] - p3[0]))

    gamma = np.arccos((l_hilf**2 + l3**2 - l2**2) / (2*l_hilf*l3))
    delta = beta - gamma

    delta_deg = delta * 180/np.pi

    plt.figure(figsize=(15, 10))
    plt.plot(alpha_deg, delta_deg)
    plt.xlabel("Winkel Platte [°]", fontsize=22)
    plt.ylabel("Winkel Abtriebshebel Servomotor [°]", size=22)
    plt.title("Umrechnung Winkel Abtriebshebel Servomotor – Winkel Platte", size=28)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlim(-15, 15)
    plt.grid()
    plt.show()

    return delta_deg



servo_umrechnung_neu(np.linspace(-12.5, 12.5, 500))