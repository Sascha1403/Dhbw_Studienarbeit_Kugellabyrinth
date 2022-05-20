# Bibliothekenimports
from collections import deque
import cv2
from imutils.video import FPS
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pigpio

from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo

# Projektimports
import ball_detection
import plate_detection
import plate_control
from devices import Webcam


def do_nothing():
    pass


# Konstante Parameter
L = 300.  # Seitelänge der Platte in mm
g_mm = 9810.  # Erdbeschleunigung in mm/s^2

# Systemparameter
pole = [-2.0, -2.0]  # 2 Wunschpolstellen
n = 0.50  # Kreisumdrehung pro Sek.
r = 200  # Radius der Kreisbaq2hn

omega = 2 * np.pi * n

activate_slider = False

# TODO: Mehrere Plots für Geschwindigkeit, Position, Stellgröße usw
# Live Graph
u_deg_list = np.zeros(shape=(100, 2))
u_deg_avg_list = np.zeros(shape=(100, 2))
v_list = np.zeros(shape=(100, 2))
frame_time_list = np.zeros(shape=100)

fig = plt.figure()
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)
ax4 = ax3.twinx()


def animate(i):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.plot(frame_time_list[-100:], v_list[-100:, 0], marker='o')
    ax2.plot(frame_time_list[-100:], v_list[-100:, 1], marker='o')
    ax3.plot(frame_time_list[-100:], np.sqrt(v_list[-100:, 0] ** 2 + v_list[-100:, 1] ** 2))

"""
# Möglichkeit der Graphischen Darstellung 
ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show(block=False)
"""




# definiere oberen und unteren Schwellwert zur Farberkennung der Farbe "blau" des Balles im HSV-Farbraum
blueLower = (105, 127, 102)
blueUpper = (115, 255, 180)

# das gleiche für orange um die Platte zu erkennen
if activate_slider:
    cv2.namedWindow("Slider")
    cv2.resizeWindow("Slider", 600, 600)
    cv2.createTrackbar("Lower H", "Slider", 0, 180, do_nothing)
    cv2.createTrackbar("Lower S", "Slider", 0, 255, do_nothing)
    cv2.createTrackbar("Lower V", "Slider", 0, 255, do_nothing)
    cv2.createTrackbar("Upper H", "Slider", 180, 180, do_nothing)
    cv2.createTrackbar("Upper S", "Slider", 255, 255, do_nothing)
    cv2.createTrackbar("Upper V", "Slider", 255, 255, do_nothing)
else:
    orangeLower = (13, 170, 127)
    orangeUpper = (17, 255, 255)

cam = Webcam(src=0, use_thread=False)
# Kurze Pause zum starten der Kamera
time.sleep(0.1)

# Hole die Regelmatrix und den Vorfilter des zugrundeliegenden Systems
R, V = plate_control.get_state_space(g_mm, pole)

# TODO: Servo Klasse

ServoX = 20
ServoY = 21

pi = pigpio.pi("192.168.0.3")

pi.set_mode(ServoX, pigpio.OUTPUT)
pi.set_mode(ServoY, pigpio.OUTPUT)

"""
pi = PiGPIOFactory(host="192.168.0.3")
ServoX = AngularServo(pin=20, min_angle=-43, max_angle=40, pin_factory=pi)
ServoY = AngularServo(pin=21, min_angle=42, max_angle=-41, pin_factory=pi)
"""

fps = FPS().start()  # Starte den FPS-Counter

start_time = frame_time_last = time.time()
ball_coords_last = None

u_deg_last = np.array([0, 0])
u_deg_avg_last = np.array([0, 0])
test_list = np.zeros(shape=1)

while True:  # Dauerschleife (bis zu einer Unterbrechung)
    if activate_slider:
        orangeLower0 = cv2.getTrackbarPos("Lower H", "Slider")
        orangeLower1 = cv2.getTrackbarPos("Lower S", "Slider")
        orangeLower2 = cv2.getTrackbarPos("Lower V", "Slider")
        orangeUpper0 = cv2.getTrackbarPos("Upper H", "Slider")
        orangeUpper1 = cv2.getTrackbarPos("Upper S", "Slider")
        orangeUpper2 = cv2.getTrackbarPos("Upper V", "Slider")

        orangeLower = (orangeLower0, orangeLower1, orangeLower2)
        orangeUpper = (orangeUpper0, orangeUpper1, orangeUpper2)

    while (time.time() - frame_time_last < 1 / 10):
        pass

    frame, _ = cam.read()
    frame_time = time.time()

    # if frame_time > frame_time_last:

    # Rauschunterdrückung durch Unschärfe und Transformation in den HSV-Farbraum
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    # blurred = cv2.blur(frame, (5, 5))
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Ballerkennung
    ball_coords = ball_detection.detect_ball(hsv, blueLower, blueUpper, show_steps=False)

    # Plattenerkennung
    plate_corners, plate_center = plate_detection.detect_plate(hsv, orangeLower, orangeUpper, show_steps=False)

    if plate_corners is not None:
        # Perspektivische Transformation (PT) der Platte
        pt_matrix = plate_detection.get_pt_matrix(plate_corners, (480, 480))  # Erstelle eine Matrix zur PT
        # TODO: Eigene if-Abfrage für die Anzeigen
        frame_pt = plate_detection.pt_image(frame, pt_matrix, (480, 480))  # PT das Ausgangsbild
        plate_center_pt = plate_detection.pt_point(plate_center, pt_matrix)

        if ball_coords is not None:
            # PT die Ballkoordinaten wenn sowohl Platten- als auch Ballkoordinaten vorhanden sind
            ball_coords_pt = plate_detection.pt_point(ball_coords, pt_matrix)
            # Ballkoordinaten relativ zur Plattenmitte
            ball_coords_pt_cnt = (ball_coords_pt[0] - plate_center_pt[0],
                                  ball_coords_pt[1] - plate_center_pt[1])

            print("------------------------------")
            print("Ballkoordinaten:     [{0:3}|{1:3}]".format(ball_coords_pt_cnt[0], ball_coords_pt_cnt[1]))

            # cv2.drawMarker(frame, ball_coords, (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=2)
            # cv2.drawMarker(frame_pt, ball_coords_pt, (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=2)

            if ball_coords_last is not None:
                if frame_time != frame_time_last:
                    ball_vel = plate_control.calc_velocity(frame_time, frame_time_last, ball_coords_pt_cnt,
                                                           ball_coords_last)
                    test_list = np.append(test_list, frame_time - frame_time_last)

                    print("Ballgeschwindigkeit: [{0:3.0f}|{1:3.0f}]".format(ball_vel[0], ball_vel[1]))

                # Zustandsgrößen
                x = np.array([ball_coords_pt_cnt[0], ball_vel[0], ball_coords_pt_cnt[1], ball_vel[1]]).T

                # Führungsgrößen
                # qw = np.array([0, 0]).T
                w = np.array([r * np.sin(omega * frame_time), r * np.cos(omega * frame_time)])
                print("Soll-Position:       [{0:3.0f}|{1:3.0f}]".format(w[0], w[1]))

                # Stellgrößen
                u_rad = np.dot(V, w) - np.dot(R, x)
                u_deg = 180 / np.pi * u_rad

                print("Stellgröße in Grad:  [{0:2.1f}|{1:2.1f}]".format(u_deg[0], u_deg[1]))

                u_deg_avg = u_deg * 0.75 + u_deg_avg_last * 0.25

                # Array mit den Stellwerten der Aktoren
                u_deg_list = np.append(u_deg_list, [u_deg], axis=0)
                u_deg_avg_list = np.append(u_deg_list, [u_deg_avg], axis=0)
                frame_time_list = np.append(frame_time_list, frame_time - start_time)
                v_list = np.append(v_list, [ball_vel], axis=0)

                u_servo_y = plate_control.servo_y_umrechnung(u_deg[1])
                u_servo_x = plate_control.servo_x_umrechnung(u_deg[0])

                # u_servo_x = plate_control.servo_umrechnung_neu(u_deg[0])
                # u_servo_y = plate_control.servo_umrechnung_neu(u_deg[1])

                pi.set_servo_pulsewidth(ServoX, u_servo_x)
                pi.set_servo_pulsewidth(ServoY, u_servo_y)

                # ServoX.angle = u_servo_x
                # ServoY.angle = u_servo_y

                u_deg_last = u_deg
                u_deg_avg_last = u_deg_avg

            ball_coords_last = ball_coords_pt_cnt

        else:  # Falls keine relativen Ballkoordinaten zur Platte gefunden werden konnten
            ball_coords_pt_cnt = None

            pi.set_servo_pulsewidth(ServoX, 1500)
            pi.set_servo_pulsewidth(ServoY, 1500)
            # ServoX.angle = 0
            # ServoY.angle = 0

        # TODO: Bug mit Markern
        # Male ein Kreuz in der Mitte der Platte
        # cv2.drawMarker(frame_pt, plate_center_pt, (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=2)
        cv2.circle(frame_pt, plate_center_pt, r, (0, 0, 0), 2)
        cv2.imshow("Perspektivische Transformation", frame_pt)

        # Male die Umrisse der Platte und ein Kreuz in der Mitte der Platte
        plate_corner_lines = plate_corners.reshape((4, 1, 2))
        cv2.polylines(frame, [plate_corner_lines], True, (0, 255, 0), 2)
        # cv2.drawMarker(frame, plate_center, (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=2)

    else:
        pi.set_servo_pulsewidth(ServoX, 1500)
        pi.set_servo_pulsewidth(ServoY, 1500)
        # ServoX.angle = 0
        # ServoY.angle = 0

    frame_time_last = frame_time

    cv2.imshow("Ausgangsbild", frame)  # Anzeigen des Ausgangsbilds in einem Fenster
    fps.update()  # FPS-Counter wird geupdated

    key = cv2.waitKey(1) & 0xFF  # Warte auf einen Tastendruck
    if key == ord("q"):  # Wenn "q" gedrückt wird, wird die Schleife beendet
        break

# stoppe den FPS-Counter und die Kamera + schließe alle Fenster
fps.stop()
cam.stop()
cv2.destroyAllWindows()

# Stoppe die Servos: erst auf die Mittelstellung fahren
pi.set_servo_pulsewidth(ServoX, 1500)
pi.set_servo_pulsewidth(ServoY, 1500)
time.sleep(1)

# Dann das PWM-Signal ausschalten
pi.set_servo_pulsewidth(ServoX, 0)
pi.set_servo_pulsewidth(ServoY, 0)

# ServoX.angle = 0
# ServoY.angle = 0


# Plot
ig, ax = plt.subplots(4, 1)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
fig.suptitle('Geschwindigkeiten', fontsize=16)
# fig.suptitle('Parameter:\n Umdrehungen pro Sekunde = %.2f \n  Polstelle= %.2f, %.2f' %(n,pole[0],pole [1]),fontsize=16)
fig.canvas.set_window_title('Geschwindigkeitsdiagramm')
# fig.canvas.set_window_title('Diagramm: Ballgeschwindigkeit')
# plt.title(label='Velocity Ball \n  Rounds per sec = %.2f \n  Polstelle= %.2f, %.2f' %(n,pole[0],pole [1]), fontdict=None, loc='center')

ax[0].set_title(r'Gesamtgeschwindigkeit', fontsize=14)
ax[0].plot(frame_time_list[-100:], np.sqrt(v_list[-100:, 0] ** 2 + v_list[-100:, 1] ** 2), label=r'$v$', marker='o')
ax[0].legend()
ax[0].grid()
ax[0].set_ylabel(r'Geschw. [px/s]', fontsize=12)
ax[0].set_ylim(0, 1000)
ax[0].set_xlabel(r'Zeit [s]', fontsize=12)
ax[0].xaxis.set_label_coords(0.9, -0.045)

ax[1].set_title(r'Geschwindigkeit in X-Richtung', fontsize=14)
ax[1].plot(frame_time_list[-100:], v_list[-100:, 0], label=r'$v_x$', color='g', marker='o')
ax[1].legend(loc="upper right")
ax[1].grid()
ax[1].set_ylabel(r'Geschw. [px/s]', fontsize=12)
ax[1].set_ylim(-1000, 1000)
ax[1].set_xlabel(r'Zeit [s]', fontsize=12)
ax[1].xaxis.set_label_coords(0.9, -0.045)

ax[2].set_title(r'Geschwindigkeit in Y-Richtung', fontsize=14)
ax[2].plot(frame_time_list[-100:], v_list[-100:, 1], label=r'$v_y$', marker='o', color='r')
ax[2].legend(loc="upper right")
ax[2].grid()
ax[2].set_ylabel(r'Geschw.  [px/s]', fontsize=12)
ax[2].set_ylim(-1000, 1000)
ax[2].set_xlabel(r'Zeit [s]', fontsize=12)
ax[2].xaxis.set_label_coords(0.9, -0.045)

ax[3].set_title(r'Zeitabstand zwischen den Frames', fontsize=14)
ax[3].plot(frame_time_list[-100:], test_list[-100:], label=r'$t$', marker='o', color='orange')
ax[3].legend(loc="upper right")
ax[3].grid()
ax[3].set_ylim(0, 0.3)
ax[3].set_ylabel(r'Zeitabstand  [s]', fontsize=12)
ax[3].set_xlabel(r'Zeit [s]', fontsize=12)
ax[3].xaxis.set_label_coords(0.9, -0.045)

plt.show()

print(test_list)

# Ausgabe der verstrichenen Zeit + durchschnittliche FPS
print("Verstrichene Zeit: {:.2f}".format(fps.elapsed()))
print("Durchschnittliche FPS: {:.2f}".format(fps.fps()))
