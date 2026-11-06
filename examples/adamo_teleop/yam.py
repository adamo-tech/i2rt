import json
import subprocess
from dataclasses import dataclass

import adamo
import numpy
from yam_ik import VELOCITY_LIMITS

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


@dataclass
class Args:
    arm_left: str | None = None
    arm_right: str | None = None
    # One stereo eye track each. Each camera has a pair of nodes; only the even one captures.
    eye_left: str = "/dev/video14"
    eye_right: str = "/dev/video6"
    # Each D405's color node: the one of its six that offers YUYV. Streamed only for an arm that runs.
    wrist_left: str = "/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_405_Intel_R__RealSense_TM__Depth_Camera_405_261923072392-video-index4"
    wrist_right: str = "/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_405_Intel_R__RealSense_TM__Depth_Camera_405_261923072394-video-index4"
    record: bool = True


NAME = "yam-san-mateo"
READY = numpy.array([0.0, 0.58, 0.57, -0.11, 0.07, 0.0])
CONTROL_PERIOD = 0.02
HOME_SPEED = VELOCITY_LIMITS / 6.0
GRIPPER_FORCE = 1.0
FRAME_RATE = 30
# The stereo pair's eye tracks. The SDK keeps them alike: both run at the lower of the two
# eyes' bitrates and restart from keyframes together.
EYE_TRACKS = (("zed_left", "stereo_left"), ("zed_right", "stereo_right"))
# 2048x1536 per eye, the largest mode both cameras offer at 30 fps, and only as MJPEG: raw
# YUYV manages 1 fps at this size. Per eye, half of what a side-by-side composite of both
# eyes was given, so the pair costs the uplink what the single track did.
EYE_WIDTH, EYE_HEIGHT, EYE_BITRATE_KBPS = 2048, 1536, 4000
# The D405's largest color mode at 30 fps.
WRIST_WIDTH, WRIST_HEIGHT, WRIST_BITRATE_KBPS = 1280, 720, 2000
# Seconds between keyframes on every track. Each is a full IDR frame, around ten times a
# normal one (the VA-API encoder has no intra refresh), and the frames behind it wait for it:
# at the default 2 s that spiked latency past 300 ms every couple of seconds.
KEYFRAME_DISTANCE = 10.0


def arm_channels(args: Args) -> list[tuple[str, str]]:
    return [(channel, side) for channel, side in ((args.arm_left, "left"), (args.arm_right, "right")) if channel]


def tailscale_ip() -> str | None:
    """This host's Tailscale IPv4 address, or None if Tailscale isn't installed or up."""
    try:
        result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return next(iter(result.stdout.split()), None)


def eye_pipeline(device: str) -> str:
    """One camera's MJPEG, decoded and converted to NV12 without leaving VA memory, for
    Adamo's native VA-API encoder to compress in place (backend="hw_pipeline").

    GStreamer stays only for the GPU decode: Adamo's own device= capture decodes MJPEG on the
    CPU. Software decode of both eyes (jpegdec / videoconvert) held 30 fps but cost ~1.5 of
    this Pentium's 4 threads, which the control loop and the arms' CAN threads then fight
    for; the VA graph costs ~0.2."""
    return (
        f"v4l2src device={device} do-timestamp=true"
        f" ! image/jpeg,width={EYE_WIDTH},height={EYE_HEIGHT},framerate={FRAME_RATE}/1 ! vajpegdec"
        f' ! vapostproc ! capsfilter caps="video/x-raw(memory:VAMemory),format=NV12,width={EYE_WIDTH},'
        f'height={EYE_HEIGHT},framerate={FRAME_RATE}/1"'
    )


def wrist_pipeline(device: str) -> str:
    """A D405's color stream as system-memory NV12. It is raw YUYV rather than MJPEG, so
    there is nothing to decode, only a conversion. That runs on the CPU: a vapostproc in a
    YUYV track aborts Adamo's hw_pipeline backend (GLib "failed to allocate" ~1e14 bytes),
    whichever memory it outputs to, while videoconvert costs ~10% of one thread per wrist."""
    return (
        f"v4l2src device={device} do-timestamp=true"
        f" ! video/x-raw,format=YUY2,width={WRIST_WIDTH},height={WRIST_HEIGHT},framerate={FRAME_RATE}/1"
        f' ! videoconvert ! capsfilter caps="video/x-raw,format=NV12,width={WRIST_WIDTH},'
        f'height={WRIST_HEIGHT},framerate={FRAME_RATE}/1"'
    )


def connect(args: Args) -> adamo.Robot:
    # Continuous rather than Episodes: Episodes keeps only time inside a task started from the
    # web app, and an operator here just takes the controls. Each arm's measured and commanded
    # joints are recorded next to the video.
    recording = args.record and adamo.Recording(
        mode=adamo.RecordingMode.Continuous(frame_rate=adamo.FrameRate.Raw()),
        topics=[f"{side}/{topic}" for _, side in arm_channels(args) for topic in ("joint_pos", "joint_cmd")],
    )
    robot = adamo.Robot(name=NAME, pairing_key="geekom", experimental=True, record=recording)
    # Direct connection details for the local viewer: the port and pairing key change every run.
    if (host := tailscale_ip()) is None:
        print(f"[{NAME}] local viewer: no Tailscale IP found (is tailscale up?)", flush=True)
    else:
        print(
            f"[{NAME}] local viewer: https://local.adamohq.com/#robot={host}:{robot.local_port}"
            f"&key={robot.local_pairing_key}",
            flush=True,
        )
    # backend="hw_pipeline" keeps GStreamer for capture but encodes with Adamo's native VA-API
    # encoder, avoiding the one-frame decode delay headsets add to vah264enc's output.
    for (track, eye), device in zip(EYE_TRACKS, (args.eye_left, args.eye_right), strict=True):
        robot.attach_video(
            track,
            pipeline=eye_pipeline(device),
            encoder="vah264enc",
            backend="hw_pipeline",
            width=EYE_WIDTH,
            height=EYE_HEIGHT,
            fps=FRAME_RATE,
            bitrate_kbps=EYE_BITRATE_KBPS,
            rate_control="vbr",
            adaptive_bitrate=True,
            keyframe_distance=KEYFRAME_DISTANCE,
            **{eye: True},
        )
    for _, side in arm_channels(args):
        robot.attach_video(
            f"wrist_{side}",
            pipeline=wrist_pipeline(getattr(args, f"wrist_{side}")),
            encoder="vah264enc",
            backend="hw_pipeline",
            width=WRIST_WIDTH,
            height=WRIST_HEIGHT,
            fps=FRAME_RATE,
            bitrate_kbps=WRIST_BITRATE_KBPS,
            rate_control="vbr",
            adaptive_bitrate=True,
            keyframe_distance=KEYFRAME_DISTANCE,
        )

    return robot


class Arm:
    def __init__(self, adamo_robot: adamo.Robot, channel: str, side: str) -> None:
        self.side = side
        self.position_publisher = adamo_robot.session.publisher(f"{NAME}/{side}/joint_pos", reliable=True)
        self.command_publisher = adamo_robot.session.publisher(f"{NAME}/{side}/joint_cmd", reliable=True)
        self.robot = get_yam_robot(channel=channel, arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310, limit_gripper_force=GRIPPER_FORCE)
        self.command = self.robot.get_joint_pos()

    def send(self) -> None:
        self.robot.command_joint_pos(self.command)
        self.position_publisher.put(json.dumps(self.robot.get_joint_pos().tolist()).encode())
        self.command_publisher.put(json.dumps(self.command.tolist()).encode())
