#!/usr/bin/python
'''
thieverized from https://gist.github.com/n3wtron/4624820
'''
import cv2
import PIL.Image as Image
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import io
import time
import threading
from snap import Vision

vision = None


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header(
                'Content-type',
                'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    img = vision.display
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    jpg = Image.fromarray(imgRGB)
                    tmpFile = io.BytesIO()
                    jpg.save(tmpFile, 'JPEG')
                    self.wfile.write(bytes("--jpgboundary", "utf-8"))
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(tmpFile.tell()))
                    self.end_headers()
                    jpg.save(self.wfile, 'JPEG')
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    break
            return
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(bytes('<html><head></head><body>', 'utf-8'))
            self.wfile.write(
                bytes('<img src="http://127.0.0.1:8080/cam.mjpg"/>',
                      'utf-8'))
            self.wfile.write(bytes('</body></html>', 'utf-8'))
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        """Handle requests in a separate thread."""


class ReaderThread(threading.Thread):
    def __init__(self):
        super().__init__()
        global vision
        self.stop = False
        vision = self.vision = Vision()
        self.vision.mode = 100

    def run(self):
        with self.vision:
            while True:
                vision.get_depths()
                vision.idepth_stats()
                vision.set_display()
                time.sleep(0.05)
                if self.stop:
                    break


def main():
    reader = ReaderThread()
    reader.start()
    try:
        server = ThreadedHTTPServer(('', 8080), CamHandler)
        print ("server started on http://localhost:8080/index.html")
        server.serve_forever()
    except KeyboardInterrupt:
        server.socket.close()
        reader.stop = True

if __name__ == '__main__':
    main()
