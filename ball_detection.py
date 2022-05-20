import cv2
import imutils
import numpy as np


def detect_ball(hsv, lower_value, higher_value, debug=False):
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

    # Falls keine Konturen gefunden werden
    center = None

    if len(contours) > 0:  # Wenn mindestens 1 Kontur gefunden wurde
        max_contour = max(contours, key=cv2.contourArea)  # Nehme die Kontur mit der größten Fläche

        # Schließe die Kontur in einem Kreis ein (alternative Mittelpunktfindung)
        #((x, y), radius) = cv2.minEnclosingCircle(max_contour)  # kleinsten umschließenden Kreis der größten Kontur
        #center = (int(x), int(y))

        # Bilde die Flächenmomente der Kontur
        moments = cv2.moments(max_contour)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))  # Schwerpunktberechnung

    return center
