import cv2
import imutils
import numpy as np
from numpy.linalg import det


def detect_plate(hsv, lower_value, higher_value, debug=False):
    mask = cv2.inRange(hsv, lower_value, higher_value)  # Maske in der Farben zwischen zwei Grenzwerten gefiltert werden
    if debug:
        cv2.imshow("Farbfilter-Maske", mask)


    # Erodieren und Erweitern (dilate) um kleines Rauschen rauszufiltern
    mask = cv2.erode(mask, None, iterations=2)
    if debug:
        cv2.imshow("Erodieren", mask)

    mask = cv2.dilate(mask, None, iterations=2)
    if debug:
        cv2.imshow("Erweitern", mask)


    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # finde Konturen in der Maske
    contours = imutils.grab_contours(contours)  # Zur Kompatibilität

    # Falls keine Platte gefunden wird
    pts_ordered = None
    center = None

    if len(contours) > 0:  # Wenn mindestens 1 Kontur gefunden wurde
        max_contour = max(contours, key=cv2.contourArea)  # Nehme die Kontur mit der größten Fläche

        # Schließe die Kontur in einem Vieleck ein (maximale Abstand der Kontur zum Vieleck = 20px)
        polygon = cv2.approxPolyDP(curve=max_contour, epsilon=20, closed=True)
        convex_polygon = cv2.convexHull(points=polygon)  # Mache das Vieleck konvex (keine Einbuchtungen nach innen)
        convex_polygon = cv2.approxPolyDP(curve=convex_polygon, epsilon=20, closed=True)

        if debug:
            # Erstelle ein Bild, male die ausgefüllte Kontur und das konvexe Vieleck und zeige es an
            contour_image = np.zeros_like(a=hsv, dtype=np.uint8)
            cv2.fillPoly(contour_image, pts=[max_contour], color=(255, 0, 0))
            cv2.drawContours(contour_image, [polygon], 0, (0, 0, 255), 4)
            cv2.drawContours(contour_image, [convex_polygon_temp], 0, (0, 255, 255), 2)
            cv2.drawContours(contour_image, [convex_polygon], 0, (0, 255, 0), 2)
            cv2.imshow("Konturen", contour_image)

        if convex_polygon.shape[0] == 4:  # Wenn es sich bei dem Vieleck um ein Viereck handelt
            pts = convex_polygon.reshape(4, 2)  # Passe die Dimensionierung an
            pts_ordered = np.zeros_like(pts)

            # Ordne die Punkte des Vierecks im Uhrzeigersinn an (oben links, oben rechts, unten rechts, unten links)
            sum = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            pts_ordered[0] = pts[np.argmin(sum)]  # Punkt oben links hat die kleinste Summe aus x und y
            pts_ordered[1] = pts[np.argmin(diff)]  # Punkt oben rechts hat die kleinste Differenz aus x und y
            pts_ordered[2] = pts[np.argmax(sum)]  # Punkt unten rechts hat die größte Summe aus x und y
            pts_ordered[3] = pts[np.argmax(diff)]  # Punkt unten links hat die größte Differenz aus x und y

            center = calc_center(pts_ordered)  # Berechne die Mitte des Rechtecks

    return pts_ordered, center


def calc_center(rect):
    # Berechnung des Mittelpunkts über 2 diagonale Geraden die jeweils 2 Punkte des Rechtsecks verbinden
    # Mittelpunkt ist am Schnittpunkt der beiden Geraden (Berechnung mit Hilfe der Determinanten)
    p1 = rect[0]
    p2 = rect[1]
    p3 = rect[2]
    p4 = rect[3]

    x_diff = (p1[0] - p3[0], p2[0] - p4[0])
    y_diff = (p1[1] - p3[1], p2[1] - p4[1])

    div = det([x_diff, y_diff])

    d = (det([p1, p3]), det([p2, p4]))

    x = det([d, x_diff]) / div
    y = det([d, y_diff]) / div
    x = int(round(x))
    y = int(round(y))

    return x, y


def get_pt_matrix(pts, pt_size):
    pts = pts.astype("float32")
    pt_x = pt_size[0]
    pt_y = pt_size[1]

    # Erstelle ein Array mit den 4 Eckpunkten des perspektivisch transformierten Bilds
    pts_pt = np.array([[0, 0], [pt_x, 0], [pt_x, pt_y], [0, pt_y]], dtype="float32")

    # Erstelle die Matrix die die perspektivische Transformation von pts zu pts_pt angibt
    matrix = cv2.getPerspectiveTransform(pts, pts_pt)

    return matrix


def pt_image(img, matrix, pt_img_size):
    # Bilde das perspektivisch transformierte (entzerrte) Bild auf Grundlage der Matrix, dem Eingangsbild und der Größe
    pt_img = cv2.warpPerspective(img, matrix, pt_img_size)
    return pt_img


def pt_point(point, matrix):
    # Passe die Dimensionierung und den Datentyp des Punkts an
    point_h = np.array([point[0], point[1], 1])

    point_pt_h = np.dot(matrix, point_h)

    point_pt = np.array([point_pt_h[0]/point_pt_h[2], point_pt_h[1]/point_pt_h[2]])
    point_pt = np.around(point_pt).astype("int")

    return point_pt


def pt_point_old(point, matrix):
    # Passe die Dimensionierung und den Datentyp des Punkts an
    point = np.array([[[point[0], point[1]]]], dtype="float32")

    # Berechne den perspektivisch transformierten Punkt auf Grundlage der Matrix und des Eingangspunkts
    point_pt = cv2.perspectiveTransform(point, matrix)
    point_pt = point_pt.reshape(2)

    # Runde und wandel zurück um in Integer
    x = int(round(point_pt[0]))
    y = int(round(point_pt[1]))

    return x, y
