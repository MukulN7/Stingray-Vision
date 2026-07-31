# Receiver Code

import gi
gi.require_version("Gst", "1.0")

from gi.repository import Gst, GLib

Gst.init(None)

pipeline = Gst.parse_launch(
    """
    udpsrc port=5000 caps=application/x-rtp,media=video,encoding-name=H264,payload=96
    ! rtph264depay
    ! avdec_h264
    ! autovideosink sync=false
    """
)

pipeline.set_state(Gst.State.PLAYING)

print("Waiting for stream...")

try:
    GLib.MainLoop().run()
except KeyboardInterrupt:
    pipeline.set_state(Gst.State.NULL)