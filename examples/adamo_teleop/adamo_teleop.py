"""Drive two YAM arms in task space from a pair of VR controllers, over Adamo.

Each arm ramps to a ready pose, then follows one hand: --arm-left the left controller,
--arm-right the right. A controller's grip is a clutch -- hold it to pin the hand to where
that arm's end-effector already is and move the arm one-for-one from there, release
to freeze. The trigger closes that arm's gripper, and Y (B on the right controller)
walks that arm back to the pose it started in.

The operator's view is the rig's two USB cameras, composited side by side into one
``stereo=True`` track. GStreamer captures and composites, Adamo's native VA-API encoder
compresses, and no frame crosses into Python.

While an operator is in control, Adamo records the video with each arm's measured
(``{side}/joint_pos``) and commanded (``{side}/joint_cmd``) joints; --no-record turns it off.

    export ADAMO_API_KEY=ak_...
    uv run examples/adamo_teleop/adamo_teleop.py --arm-left can1 --arm-right can0 \
        --eye-left /dev/video6 --eye-right /dev/video4
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
    # Composited into one stereo track. Each camera has a pair of nodes; only the even one captures.
    eye_left: str = "/dev/video6"
    eye_right: str = "/dev/video4"
    record: bool = True


args = tyro.cli(Args)
NAME = "yam-san-mateo"
ARMS = [(ch, side) for ch, side in ((args.arm_left, "left"), (args.arm_right, "right")) if ch]
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
# 1080p per eye, a mode both cameras offer. The composite is what has to fit the
# 4096-pixel width an H.264 encoder (and a headset decoder) will take, so this is
# 3840 wide on the wire.
EYE_WIDTH, EYE_HEIGHT, FPS = 1920, 1080, 30
BITRATE_KBPS = 8000

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


def eye_branch(device: str, sink: str) -> str:
    """One camera, decoded and converted, into one compositor pad. The queue is one deep
    and leaky, so an eye that stalls drops its own frames rather than holding the other."""
    return (
        f"v4l2src device={device} do-timestamp=true"
        f" ! image/jpeg,width={EYE_WIDTH},height={EYE_HEIGHT},framerate={FPS}/1 ! jpegdec"
        " ! videoconvert ! video/x-raw,format=NV12"
        " ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
        f" ! {sink}"
    )


# The composite's caps are a named, quoted capsfilter: the parser reads a trailing bare
# caps greedily and would run straight past the space into the next chain.
PIPELINE = "  ".join(
    (
        f"compositor name=sbs background=black sink_0::xpos=0 sink_1::xpos={EYE_WIDTH}"
        f' ! capsfilter caps="video/x-raw,format=NV12,width={2 * EYE_WIDTH},'
        f'height={EYE_HEIGHT},framerate={FPS}/1"',
        eye_branch(args.eye_left, "sbs.sink_0"),
        eye_branch(args.eye_right, "sbs.sink_1"),
    )
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
# Continuous rather than Episodes: Episodes keeps only time inside a task started from the
# web app, and an operator here just takes the controls.
recording = args.record and adamo.Recording(mode=adamo.RecordingMode.Continuous(frame_rate=adamo.FrameRate.Raw()))
adamo_robot = adamo.Robot(api_key=os.environ["ADAMO_API_KEY"], name=NAME, record=recording, experimental=True)
# backend="hw_pipeline" keeps GStreamer for capture but encodes with Adamo's native VA-API
# encoder, avoiding the one-frame decode delay headsets add to vah264enc's output.
adamo_robot.attach_video(
    TRACK,
    pipeline=PIPELINE,
    encoder="vah264enc",
    backend="hw_pipeline",
    width=2 * EYE_WIDTH,
    height=EYE_HEIGHT,
    fps=FPS,
    bitrate_kbps=BITRATE_KBPS,
    rate_control="vbr",
    adaptive_bitrate=True,
    stereo=True,
)
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
