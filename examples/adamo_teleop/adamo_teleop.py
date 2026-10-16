"""Drive two YAM arms in task space from a pair of VR controllers, over Adamo.

Each arm ramps to a ready pose, then follows one hand: --arm-left the left controller,
--arm-right the right. A controller's grip is a clutch -- hold it to pin the hand to where
that arm's end-effector already is and move the arm one-for-one from there, release
to freeze. The trigger closes that arm's gripper, and Y (B on the right controller)
walks that arm back to the pose it started in.

The operator's view is a ZED Mini, read as a plain UVC camera: it delivers both eyes
side by side in one frame, which goes out as one ``stereo=True`` track. GStreamer captures,
Adamo's native VA-API encoder compresses, no frame crosses into Python, and no ZED SDK is
needed.

While an operator runs a task from the web app, Adamo records the video with each arm's measured
(``{side}/joint_pos``) and commanded (``{side}/joint_cmd``) joints; --no-record turns it off.

    export ADAMO_API_KEY=ak_...
    uv run examples/adamo_teleop/adamo_teleop.py --arm-left can1 --arm-right can0 --zed /dev/video2
"""

import json
import os
import threading
from dataclasses import dataclass

import adamo
import jaxlie
import numpy as np
import tyro
from adamo.xr import Joy, PoseStamped, XrController, XrFrame, subscribe_xr_control
from yam_ik import VELOCITY_LIMITS, YamIK

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
# Where the arms settle when unpowered, read off both arms and averaged, so cutting
# power from the ready pose doesn't drop them.
READY = np.array([0.0, 0.58, 0.57, -0.11, 0.07, 0.0])
DT = 0.02
# Rehoming runs a sixth of the speed the IK clamps hand-tracking to: nobody is steering it,
# so it should look deliberate rather than as quick as the arm can manage.
HOME_SPEED = VELOCITY_LIMITS / 6.0
# Newtons the gripper holds once it closes on something: a tenth of the library's 50 N default.
GRIPPER_FORCE = 1.0

TRACK = "zed"
# The ZED Mini's 1080p mode: 1920x1080 per eye, 3840x1080 side by side, left eye on the
# left. Its only larger mode (2208x1242 per eye, 4416 wide, 15 fps) is past the 4096-pixel
# width an H.264 encoder (and a headset decoder) will take. The camera sends raw YUYV only,
# ~250 MB/s at this mode, so it has to sit on a USB 3 port. GStreamer drops corrupt USB
# buffers and repeats intact frames across gaps, so 30 FPS out is not 30 fresh frames.
WIDTH, HEIGHT, FPS = 3840, 1080, 30
BITRATE_KBPS = 8000
WRIST_WIDTH, WRIST_HEIGHT = 1280, 720
WRIST_BITRATE_KBPS = 2000

# WebXR local-floor (+x right, +y up, -z forward) -> arm world (+x forward, +y left, +z up).
XR2W = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

lock = threading.Lock()
state = {side: {"pos": None, "rot": None, "clutch": False, "grip": None, "home": False} for _, side in ARMS}


def quat_to_mat(q: object) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def on_xr(sample: object) -> None:
    # The headset sends either one XrFrame on json/xr/frame, or one message per device on
    # json/xr/{left,right}/{pose,joy}; both reduce to (side, pose, joy) updates.
    msg = sample.message
    if isinstance(msg, XrFrame):
        updates = [(side, c.pose, c) for side in state if (c := getattr(msg, side)) is not None]
    else:
        side, kind = sample.channel.split("/")[-2:]
        updates = [(side, msg if kind == "pose" and isinstance(msg, PoseStamped) else None, msg)]
    with lock:
        for name, pose, joy in updates:
            if name not in state:
                continue
            side = state[name]
            if pose is not None:
                side["pos"] = XR2W @ np.array([pose.position.x, pose.position.y, pose.position.z])
                side["rot"] = XR2W @ quat_to_mat(pose.orientation) @ XR2W.T
            if not isinstance(joy, (Joy, XrController)):
                continue
            side["clutch"] = len(joy.buttons) > 1 and bool(joy.buttons[1])
            # Y on the left controller, B on the right: the same index in the Touch mapping.
            side["home"] = len(joy.buttons) > 5 and bool(joy.buttons[5])
            trigger = (
                float(joy.axes[4])
                if len(joy.axes) > 4 and joy.axes[4] > 0
                else float(bool(joy.buttons and joy.buttons[0]))
            )
            if trigger > 0.05:  # a released trigger leaves the gripper where it was
                side["grip"] = 1.0 - trigger


# Filter corruption before conversion: v4l2src can emit full-sized buffers marked corrupted
# even when only part of the USB frame arrived. The pipeline ends in raw NV12; Adamo adds
# the encoder.
PIPELINE = (
    f"v4l2src device={args.zed} io-mode=mmap"
    f' ! capsfilter caps="video/x-raw,format=YUY2,width={WIDTH},height={HEIGHT},framerate={FPS}/1"'
    " ! identity drop-buffer-flags=corrupted"
    " ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream"
    " ! videoconvert n-threads=2"
    " ! videorate"
    f' ! capsfilter caps="video/x-raw,format=NV12,framerate={FPS}/1"'
)

ik = YamIK()


class Arm:
    """One robot, the hand it follows, and where its end-effector is being asked to go."""

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
        self.q = READY.copy()
        position, wxyz = ik.tcp_pose(self.q)
        self.target, self.rot = position, np.asarray(jaxlie.SO3(wxyz).as_matrix())
        self.anchor = None
        self.homing = False

    def step(self) -> None:
        with lock:
            s = state[self.side]
            hand, hand_rot, clutch, grip, home = s["pos"], s["rot"], s["clutch"], s["grip"], s["home"]
        tracking = clutch and hand is not None
        if home and not tracking and not self.homing:  # a held clutch is never overridden
            self.homing = True
            print(f"[teleop] {self.side} rehoming to {np.round(READY, 3)}", flush=True)

        if tracking:
            self.homing = False
            if self.anchor is None:
                self.anchor = (hand, hand_rot, self.target, self.rot)
                print(f"[teleop] {self.side} grip held, tracking from {np.round(self.target, 4)}", flush=True)
            hand0, hand_rot0, ee0, ee_rot0 = self.anchor
            self.target, self.rot = ee0 + (hand - hand0), hand_rot @ hand_rot0.T @ ee_rot0
            # The solve is warm-started from the last configuration and rate-limited inside, so an
            # unreachable target degrades into a lagging arm rather than a jump, and recovers.
            wxyz = np.asarray(jaxlie.SO3.from_matrix(self.rot).wxyz)
            self.q, _, _ = ik.solve(self.target, wxyz, self.q, DT)
            self.cmd[:6] = self.q
        else:
            if self.anchor is not None:
                self.anchor = None
                print(f"[teleop] {self.side} grip released, holding {np.round(self.target, 4)}", flush=True)
            if self.homing:
                # Joint space, at HOME_SPEED. The target is carried along so a re-clutch
                # mid-rehome anchors to where the arm actually is.
                self.q = self.q + np.clip(READY - self.q, -HOME_SPEED * DT, HOME_SPEED * DT)
                position, wxyz = ik.tcp_pose(self.q)
                self.target, self.rot = position, np.asarray(jaxlie.SO3(wxyz).as_matrix())
                self.cmd[:6] = self.q
                if np.array_equal(self.q, READY):
                    self.homing = False
                    print(f"[teleop] {self.side} rehomed", flush=True)
        if grip is not None:
            self.cmd[6] = grip
        self.robot.command_joint_pos(self.cmd)
        self.pos_pub.put(json.dumps(self.robot.get_joint_pos().tolist()).encode())
        self.cmd_pub.put(json.dumps(self.cmd.tolist()).encode())


# Compile the solver before the motors are enabled: the first solve takes seconds (and can
# starve the CAN threads), every one after it a few milliseconds.
ik.solve(*ik.tcp_pose(READY), READY, DT)
# Episodes keeps only the time inside a task started from the web app, one file per task.
recording = args.record and adamo.Recording(mode=adamo.RecordingMode.Episodes(frame_rate=FPS), topics=[f"{side}/{topic}" for _, side in ARMS for topic in ("joint_pos", "joint_cmd")])
adamo_robot = adamo.Robot(api_key=os.environ["ADAMO_API_KEY"], name=NAME, record=recording)
# backend="hw_pipeline" keeps GStreamer for capture but encodes with Adamo's native VA-API
# encoder, avoiding the one-frame decode delay headsets add to vah264enc's output.
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
subscribe_xr_control(adamo_robot, on_xr, max_age=0.25)
# attach_video only queues the graph; run() is what starts the Rust pipeline, and it
# blocks -- so it goes in a daemon thread and the control loop keeps the main one.
video = threading.Thread(target=adamo_robot.run, daemon=True)
video.start()
arms = [Arm(channel, side) for channel, side in ARMS]
print(f"[teleop] {NAME}: grip to move an arm, trigger for its gripper", flush=True)

try:
    # Ramp to the ready pose: the home pose sits on joint2/joint3's lower bound.
    starts = [arm.cmd[:6].copy() for arm in arms]
    rate = adamo.Rate(1 / DT)
    for i in range(1, 201):
        for arm, q0 in zip(arms, starts, strict=True):
            arm.cmd[:6] = q0 + (READY - q0) * (i / 200)
            arm.robot.command_joint_pos(arm.cmd)
        rate.sleep()

    # Rate keeps a fixed schedule, so the milliseconds the solve costs don't stretch DT,
    # which is what the IK's velocity clamp is scaled by.
    while video.is_alive():  # an operator who can't see shouldn't be steering
        for arm in arms:
            arm.step()
        rate.sleep()
    print("[teleop] video stopped", flush=True)
except KeyboardInterrupt:
    print("\n[teleop] interrupted", flush=True)
finally:
    for arm in arms:
        arm.robot.close()
    adamo_robot.close()
