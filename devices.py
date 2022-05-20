from threading import Thread
import cv2
import time


class Webcam:
    def __init__(self, src=0, use_thread=False):
        # TODO: Kameraparameter (FPS, Auflösung) veränderbar machen
        # Initialisiere den Video-Stream von openCV und lese den ersten Frame ein
        self.stream = cv2.VideoCapture(src)
        _, self.frame = self.stream.read()  # Unterstrich (_) bedeutet, dass die Varable nicht benutzt wird

        self.use_thread = use_thread
        self.frame_time = None          # Zeitstempel des neuesten Frames
        self.stopped = False            # Initialisiere eine Variable, um den Thread zu stoppen

        # Stelle die FPS (Parameter 5) der Kamera ein
        #self.stream.set(5, 30)
        #print(self.stream.get(5))

        # Wenn angegeben, starte einen extra Thread, um die Frames der Webcam zu holen
        if self.use_thread:
            t = Thread(target=self.update, name="Webcam-Thread", args=())
            t.daemon = True
            t.start()

    def update(self):
        # Dauerschleife um die neuesten Kamerabilder zu holen
        while True:
            # Wenn die Stopp-Variable gesetzt ist, beende die Dauerschleife
            if self.stopped:
                return

            # Lese den neuesten Frame ein und setze den Zeitstempel auf die aktuelle Zeit
            _, self.frame = self.stream.read()
            self.frame_time = time.time()

    def read(self):
        if not self.use_thread:
            _, self.frame = self.stream.read()
            self.frame_time = time.time()

        # Gebe den neuesten Frame und die Frame-Time zurück
        return self.frame, self.frame_time

    def stop(self):
        # Stoppe den Thread
        self.stopped = True
