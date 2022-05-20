# Bibliothekenimports
import cv2  # Bildvearbeitung
from imutils.video import FPS  # FPS-Zähler
import time  # Zeitmodul (Zeit messen)
import numpy as np  # no words needed
import matplotlib.pyplot as plt  # Graphen anzeigen/plotten
import pigpio  # Ansteuerung der Raspberry Pi Pins
import win32api, win32process, win32con  # Windows API zur Festlegung der Prozesspriorität

# Projektimports
import ball_detection
import plate_detection
import plate_control
from devices import Webcam


def do_nothing():
    pass


# Setze Prozesspriorität auf "Real-Time"
process_id = win32api.GetCurrentProcessId()
handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, process_id)
win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)

# Konstante Parameter
L = 300.  # Seitelänge der Platte in mm
g_mm = 9810.  # Erdbeschleunigung in mm/s^2

# Systemparameter
pole = [-2.0, -2.0]  # 2 Wunschpolstellen
n = 0.5  # Kreisumdrehung pro Sek.
r = 200  # Radius der Kreisbaq2hn

omega = 2 * np.pi * n

activate_slider = False

# TODO: Mehrere Plots für Geschwindigkeit, Position, Stellgröße usw


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

# Starte die Kamera
cam = Webcam(src=0, use_thread=False)
# Kurze Pause zum starten der Kamera
time.sleep(0.1)

# Hole die Regelmatrix und den Vorfilter des zugrundeliegenden Systems
R, V = plate_control.get_state_space(g_mm, pole)

ServoX = 20
ServoY = 21

pi = pigpio.pi(host="192.168.0.3")

pi.set_mode(ServoX, pigpio.OUTPUT)
pi.set_mode(ServoY, pigpio.OUTPUT)

fps = FPS().start()  # Starte den FPS-Counter

frame_time_last = time.time()
ball_coords_last = None

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

    # Neuer Frame soll erst 0.1s nach dem letzten aufgenommen werden (fixe 10 FPS)
    while time.time() - frame_time_last < 0.1:
        pass
    frame, frame_time = cam.read()

    # Rauschunterdrückung durch Unschärfe und Transformation in den HSV-Farbraum
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Ball- und Plattenerkennung
    ball_coords = ball_detection.detect_ball(hsv, blueLower, blueUpper, debug=False)
    plate_corners, plate_center = plate_detection.detect_plate(hsv, orangeLower, orangeUpper, debug=False)

    if plate_corners is not None:
        # Perspektivische Transformation (PT) der Platte
        pt_matrix = plate_detection.get_pt_matrix(plate_corners, (480, 480))  # Erstelle eine Matrix zur PT

        # Anzeige
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

            # Anzeige
            cv2.drawMarker(frame, ball_coords, (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=2)
            cv2.drawMarker(frame_pt, ball_coords_pt, (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=2)

            if ball_coords_last is not None:
                # Geschwindigkeit
                if frame_time != frame_time_last:
                    v = plate_control.calc_velocity(frame_time, frame_time_last, ball_coords_pt_cnt, ball_coords_last)

                print("Ballgeschwindigkeit: [{0:3.0f}|{1:3.0f}]".format(v[0], v[1]))

                # Zustandsgrößen
                x = np.array([ball_coords_pt_cnt[0], v[0], ball_coords_pt_cnt[1], v[1]]).T

                # Führungsgrößen (Mittelpunkt oder Kreisbahn)
                # w = np.array([0, 0]).T
                w = np.array([r * np.sin(omega * frame_time), r * np.cos(omega * frame_time)])
                print("Soll-Position:       [{0:3.0f}|{1:3.0f}]".format(w[0], w[1]))

                # Stellgrößen
                u_rad = np.dot(V, w) - np.dot(R, x)
                u_deg = 180 / np.pi * u_rad

                print("Stellgröße in Grad:  [{0:2.1f}|{1:2.1f}]".format(u_deg[0], u_deg[1]))

                # Rechne den Winkel der Platte in die Impulsdauer der Servos um
                u_servo_x = plate_control.servo_y_umrechnung(-u_deg[0])
                u_servo_y = plate_control.servo_x_umrechnung(-u_deg[1])
                pi.set_servo_pulsewidth(ServoX, u_servo_x)
                pi.set_servo_pulsewidth(ServoY, u_servo_y)


        else:  # Falls keine relativen Ballkoordinaten zur Platte gefunden werden konnten
            ball_coords_pt_cnt = None

            pi.set_servo_pulsewidth(ServoX, 1500)
            pi.set_servo_pulsewidth(ServoY, 1500)

        # Anzeige
        cv2.drawMarker(frame_pt, plate_center_pt, (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=2)
        cv2.circle(frame_pt, plate_center_pt, r, (255, 255, 255), 2)
        cv2.imshow("Perspektivische Transformation", frame_pt)
        plate_corner_lines = plate_corners.reshape((4, 1, 2))
        cv2.polylines(frame, [plate_corner_lines], True, (0, 255, 0), 2)
        cv2.drawMarker(frame, plate_center, (0, 255, 0), cv2.MARKER_CROSS, markerSize=15, thickness=2)

        ball_coords_last = ball_coords_pt_cnt


    else:  # Wenn keine Platte erkannt wurde
        pi.set_servo_pulsewidth(ServoX, 1500)
        pi.set_servo_pulsewidth(ServoY, 1500)

    frame_time_last = frame_time  # Speichere die Frame Time für den nächsten Schleifendurchlauf

    # Anzeige
    cv2.imshow("Ausgangsbild", frame)

    fps.update()

    # Schließe das Fenster beim Drücken von der Taste "q"
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# stoppe den FPS-Counter und die Kamera und schließe alle Fenster
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

# Ausgabe der verstrichenen Zeit + durchschnittliche FPS
print("Verstrichene Zeit: {:.2f}s".format(fps.elapsed()))
print("Durchschnittliche FPS: {:.2f}FPS".format(fps.fps()))
