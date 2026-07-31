import cv2
import threading
import time


class CameraStreamer:
    def __init__(self, pipeline, cam_name):
        self.pipeline = pipeline
        self.cam_name = cam_name

        self.cap = None
        self.frame = None

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def start(self):
        for _ in range(5):
            self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

            if self.cap.isOpened():
                print(f"{self.cam_name}: Connected")
                return True

            print(f"{self.cam_name}: Retrying...")
            time.sleep(2)

        print(f"{self.cam_name}: Failed to connect")
        return False

    def run(self):
        if not self.start():
            return

        while not self.stop_event.is_set():

            ret, frame = self.cap.read()

            if not ret:
                print(f"{self.cam_name}: Frame not received")
                continue

            with self.lock:
                self.frame = frame

        self.close()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stop_event.set()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print(f"{self.cam_name}: Closed")


pipeline1 = (
    'udpsrc port=5000 '
    'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! '
    'rtpjitterbuffer ! '
    'rtph264depay ! '
    'avdec_h264 ! '
    'videoconvert ! '
    'appsink drop=true sync=false'
)

pipeline2 = (
    'udpsrc port=5001 '
    'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! '
    'rtpjitterbuffer ! '
    'rtph264depay ! '
    'avdec_h264 ! '
    'videoconvert ! '
    'appsink drop=true sync=false'
)

print(cv2.getBuildInformation())
camera1 = CameraStreamer(pipeline1, "Camera1")
camera2 = CameraStreamer(pipeline2, "Camera2")

t1 = threading.Thread(target=camera1.run)
t2 = threading.Thread(target=camera2.run)

t1.start()
t2.start()

try:
    while True:

        frame = camera1.get_frame()
        if frame is not None:
            cv2.imshow("Camera1", frame)

        frame = camera2.get_frame()
        if frame is not None:
            cv2.imshow("Camera2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    camera1.stop()
    camera2.stop()

    t1.join()
    t2.join()

    cv2.destroyAllWindows()