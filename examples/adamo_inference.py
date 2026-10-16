"""Let a cloud policy drive two YAM arms in joint space, over Adamo.

The robot is the one adamo_teleop.py runs: the same name, the same ``zed`` and wrist tracks,
and the same measured (``{side}/joint_pos``) and commanded (``{side}/joint_cmd``) joint topics.
Instead of following VR controllers, each arm follows the joint targets a policy publishes on
its control channel ``{side}/joint_cmd``, as ``{"action": [six joints, gripper]}``.

Each arm walks to the ready pose, then moves toward the newest target no faster than teleop
lets an arm move. When targets stop arriving it walks back to the ready pose, and ctrl-c does
the same before the motors are released.

    export ADAMO_API_KEY=ak_...
    uv run examples/adamo_inference.py --arm-left can1 --arm-right can0 --zed /dev/video2
"""

import json
import os
import signal
import threading
import time
from dataclasses import dataclass

import adamo
import numpy as np
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


@dataclass
class Args:
    """Name the CAN interface of each arm to run: --arm-left, --arm-right, or both."""

    arm_left: str | None = None
    arm_right: str | None = None
    # The ZED Mini's capture node; the odd node after it is metadata only.
    zed: str = "/dev/video2"
    wrist_left: str | None = None
    wrist_right: str | None = None
    record: bool = True


args = tyro.cli(Args)
NAME = "yam-london"
ARMS = [(ch, side) for ch, side in ((args.arm_left, "left"), (args.arm_right, "right")) if ch]
WRIST_CAMERAS = [(device, side) for device, side in ((args.wrist_left, "left"), (args.wrist_right, "right")) if device]
# Where the arms settle when unpowered, so cutting power from the ready pose doesn't drop them.
READY = np.array([0.0, 0.58, 0.57, -0.11, 0.07, 0.0])
DT = 0.02
# The clamp teleop's IK puts on hand tracking (yam_ik.VELOCITY_LIMITS), in rad/s.
VELOCITY_LIMITS = np.array([2.5, 2.5, 2.5, 4.0, 4.0, 4.0])
# Walking home runs at a sixth of that: nobody is steering it.
HOME_SPEED = VELOCITY_LIMITS / 6.0
# A target older than this is not followed: the policy sends one every frame.
TARGET_TIMEOUT = 0.25
# Newtons the gripper holds once it closes on something: a tenth of the library's 50 N default.
GRIPPER_FORCE = 1.0

TRACK = "zed"
WIDTH, HEIGHT, FPS = 3840, 1080, 30
BITRATE_KBPS = 8000
WRIST_WIDTH, WRIST_HEIGHT = 1280, 720
WRIST_BITRATE_KBPS = 2000

# Same capture graph as adamo_teleop.py; see there for why each element is in it.
PIPELINE = (
    f"v4l2src device={args.zed} io-mode=mmap"
    f' ! capsfilter caps="video/x-raw,format=YUY2,width={WIDTH},height={HEIGHT},framerate={FPS}/1"'
    " ! identity drop-buffer-flags=corrupted"
    " ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream"
    " ! videoconvert n-threads=2"
    " ! videorate"
    f' ! capsfilter caps="video/x-raw,format=NV12,framerate={FPS}/1"'
)

# The newest target of each arm and when it arrived.
targets = {side: None for _, side in ARMS}
stopping = threading.Event()


def on_target(side: str, sample: object) -> None:
    targets[side] = (time.monotonic(), np.asarray(sample.message["action"], dtype=float))


class Arm:
    """One robot and the joints it is being asked to hold."""

    def __init__(self, channel: str, side: str) -> None:
        self.side = side
        # The state and action a recording files next to each frame.
        self.pos_pub = adamo_robot.session.publisher(f"{NAME}/{side}/joint_pos", reliable=True)
        self.cmd_pub = adamo_robot.session.publisher(f"{NAME}/{side}/joint_cmd", reliable=True)
        self.robot = get_yam_robot(
            channel=channel,
            arm_type=ArmType.YAM,
            gripper_type=GripperType.LINEAR_4310,
            limit_gripper_force=GRIPPER_FORCE,
        )
        self.cmd = self.robot.get_joint_pos()
        self.following = False

    def step(self, follow: bool) -> None:
        target = targets[self.side]
        following = follow and target is not None and time.monotonic() - target[0] < TARGET_TIMEOUT
        if following != self.following:
            self.following = following
            print(f"[inference] {self.side} {'following the policy' if following else 'walking to the ready pose'}", flush=True)

        if following:
            action = target[1]
            self.cmd[:6] += np.clip(action[:6] - self.cmd[:6], -VELOCITY_LIMITS * DT, VELOCITY_LIMITS * DT)
            self.cmd[6] = np.clip(action[6], 0.0, 1.0)
        else:
            self.cmd[:6] += np.clip(READY - self.cmd[:6], -HOME_SPEED * DT, HOME_SPEED * DT)
        self.robot.command_joint_pos(self.cmd)
        self.pos_pub.put(json.dumps(self.robot.get_joint_pos().tolist()).encode())
        self.cmd_pub.put(json.dumps(self.cmd.tolist()).encode())


# Episodes keeps only the time inside a task started from the web app, one file per task.
recording = args.record and adamo.Recording(mode=adamo.RecordingMode.Episodes(frame_rate=FPS), topics=[f"{side}/{topic}" for _, side in ARMS for topic in ("joint_pos", "joint_cmd")])
adamo_robot = adamo.Robot(api_key=os.environ["ADAMO_API_KEY"], name=NAME, record=recording)
adamo_robot.attach_video(
    TRACK,
    pipeline=PIPELINE,
    encoder="vah264enc",
    backend="hw_pipeline",
    width=WIDTH,
    height=HEIGHT,
    fps=FPS,
    bitrate_kbps=BITRATE_KBPS,
    rate_control="vbr",
    adaptive_bitrate=False,
    stereo=True,
)
for device, side in WRIST_CAMERAS:
    adamo_robot.attach_video(f"wrist_{side}", pipeline=f'v4l2src device={device} ! capsfilter caps="image/jpeg,width={WRIST_WIDTH},height={WRIST_HEIGHT},framerate={FPS}/1" ! jpegdec ! videoconvert ! capsfilter caps="video/x-raw,format=NV12"', encoder="vah264enc", backend="hw_pipeline", width=WRIST_WIDTH, height=WRIST_HEIGHT, fps=FPS, bitrate_kbps=WRIST_BITRATE_KBPS, rate_control="vbr", adaptive_bitrate=False)
for _, side in ARMS:
    adamo_robot.on(f"{side}/joint_cmd", lambda sample, side=side: on_target(side, sample), max_age=TARGET_TIMEOUT)
# attach_video only queues the graph; run() is what starts the Rust pipeline, and it
# blocks -- so it goes in a daemon thread and the control loop keeps the main one.
video = threading.Thread(target=adamo_robot.run, daemon=True)
video.start()
arms = [Arm(channel, side) for channel, side in ARMS]
# A handler rather than KeyboardInterrupt, so the arms can walk home before the motors go slack.
signal.signal(signal.SIGINT, lambda *_: stopping.set())
signal.signal(signal.SIGTERM, lambda *_: stopping.set())
print(f"[inference] {NAME}: following joint targets on {{side}}/joint_cmd", flush=True)

try:
    rate = adamo.Rate(1 / DT)
    # A policy that can't see shouldn't be steering. An arm whose motor thread died is limp,
    # and nothing else here would notice: the commands below would go nowhere.
    while video.is_alive() and not stopping.is_set() and all(arm.robot._server_thread.is_alive() for arm in arms):
        for arm in arms:
            arm.step(follow=True)
        rate.sleep()

    print("[inference] stopping: walking to the ready pose", flush=True)
    while not all(np.array_equal(arm.cmd[:6], READY) for arm in arms):
        for arm in arms:
            arm.step(follow=False)
        rate.sleep()
    time.sleep(1.0)  # let the arms catch up with the last command
finally:
    for arm in arms:
        arm.robot.close()
    adamo_robot.close()

os._exit(0)  # adamo's native side can abort while the interpreter exits
