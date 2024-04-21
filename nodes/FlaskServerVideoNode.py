from flask import Flask, render_template, Response
from threading import Thread
import numpy as np
import cv2


class EndpointAction(object):

    def __init__(self, action):
        self.action = action

    def __call__(self, *args):
        result = self.action()
        response = Response(result, status=200, headers={})
        return response


class VideoServer(object):
    app = None
    def __init__(self, index_page: str, host_ip: str, template_folder: str):
        self.app = Flask(__name__, template_folder=template_folder)
        self.app.add_url_rule('/', 'index', EndpointAction(self._index))
        self.app.add_url_rule('/video', 'video', self._update_page)

        self.host_ip = host_ip
        self.index_page = index_page
        self._frame = np.zeros(shape=(640, 480), dtype=np.uint8)

    def _index(self) -> str:
        return render_template(self.index_page)
  
    def _gen(self):
        while True:
            ret, jpeg = cv2.imencode('.jpg', self._frame)
            encoded_image = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + encoded_image + b'\r\n\r\n')
            
    def _update_page(self) -> Response:
        return Response(self._gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def update_image(self, image: np.array):
        self._frame = image

    def run(self):
        self.app_thread = Thread(target=self.app.run, args=(self.host_ip, 8100))
        self.app_thread.start()




if __name__ == "__main__":
    video_server = VideoServer("index.html", "0.0.0.0", "../utils_local/templates")
    video_server.run()
    while True:
        img = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        video_server.update_image(img)
