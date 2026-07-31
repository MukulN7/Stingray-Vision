# Source Device Code

import gi
gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib

Gst.init(None)

RECEIVER_IP = "192.168.1.100"

pipeline = Gst.parse_launch(
    f"""
    v4l2src device=/dev/video0
    ! video/x-raw,width=1280,height=720,framerate=30/1
    ! videoconvert
    ! x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast
    ! rtph264pay pt=96
    ! udpsink host={RECEIVER_IP} port=5000
    """
)

pipeline.set_state(Gst.State.PLAYING)

print("Streaming... Press Ctrl+C to stop.")

try:
    GLib.MainLoop().run()
except KeyboardInterrupt:
    pipeline.set_state(Gst.State.NULL)