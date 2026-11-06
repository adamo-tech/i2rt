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
    zed: str = "/dev/video2"
    wrist_left: str | None = None
    wrist_right: str | None = None
    record: bool = True


NAME = "yam-san-mateo"
READY = numpy.array([0.0, 0.58, 0.57, -0.11, 0.07, 0.0])
CONTROL_PERIOD = 0.02
HOME_SPEED = VELOCITY_LIMITS / 6.0
GRIPPER_FORCE = 1.0
FRAME_RATE = 30
ZED_WIDTH, ZED_HEIGHT, ZED_BITRATE_KBPS = 3840, 1080, 8000
WRIST_WIDTH, WRIST_HEIGHT, WRIST_BITRATE_KBPS = 1280, 720, 2000


def arm_channels(args: Args) -> list[tuple[str, str]]:
    return [(channel, side) for channel, side in ((args.arm_left, "left"), (args.arm_right, "right")) if channel]


def tailscale_ip() -> str | None:
    """This host's Tailscale IPv4 address, or None if Tailscale isn't installed or up."""
    try:
        result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return next(iter(result.stdout.split()), None)


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
    robot.attach_video("zed", pipeline=f'v4l2src device={args.zed} io-mode=mmap ! capsfilter caps="video/x-raw,format=YUY2,width={ZED_WIDTH},height={ZED_HEIGHT},framerate={FRAME_RATE}/1" ! identity drop-buffer-flags=corrupted ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! videoconvert n-threads=2 ! videorate ! capsfilter caps="video/x-raw,format=NV12,framerate={FRAME_RATE}/1"', encoder="vah264enc", backend="hw_pipeline", width=ZED_WIDTH, height=ZED_HEIGHT, fps=FRAME_RATE, bitrate_kbps=ZED_BITRATE_KBPS, rate_control="vbr", adaptive_bitrate=False, stereo=True)

    for device, side in ((args.wrist_left, "left"), (args.wrist_right, "right")):
        if device:
            robot.attach_video(f"wrist_{side}", pipeline=f'v4l2src device={device} ! capsfilter caps="image/jpeg,width={WRIST_WIDTH},height={WRIST_HEIGHT},framerate={FRAME_RATE}/1" ! jpegdec ! videoconvert ! capsfilter caps="video/x-raw,format=NV12"', encoder="vah264enc", backend="hw_pipeline", width=WRIST_WIDTH, height=WRIST_HEIGHT, fps=FRAME_RATE, bitrate_kbps=WRIST_BITRATE_KBPS, rate_control="vbr", adaptive_bitrate=False)

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
