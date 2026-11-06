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

import threading

import adamo
import jaxlie
import numpy as np
import tyro
from adamo.xr import Joy, PoseStamped, XrController, XrFrame, subscribe_xr_control
from yam import CONTROL_PERIOD, HOME_SPEED, NAME, READY, Args, Arm, arm_channels, connect
from yam_ik import YamIK

args = tyro.cli(Args)
ARMS = arm_channels(args)

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


ik = YamIK()


class TeleopArm(Arm):
    """One robot, the hand it follows, and where its end-effector is being asked to go."""

    def __init__(self, adamo_robot: adamo.Robot, channel: str, side: str) -> None:
        super().__init__(adamo_robot, channel, side)
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
            self.q, _, _ = ik.solve(self.target, wxyz, self.q, CONTROL_PERIOD)
            self.command[:6] = self.q
        else:
            if self.anchor is not None:
                self.anchor = None
                print(f"[teleop] {self.side} grip released, holding {np.round(self.target, 4)}", flush=True)
            if self.homing:
                # Joint space, at HOME_SPEED. The target is carried along so a re-clutch
                # mid-rehome anchors to where the arm actually is.
                self.q = self.q + np.clip(READY - self.q, -HOME_SPEED * CONTROL_PERIOD, HOME_SPEED * CONTROL_PERIOD)
                position, wxyz = ik.tcp_pose(self.q)
                self.target, self.rot = position, np.asarray(jaxlie.SO3(wxyz).as_matrix())
                self.command[:6] = self.q
                if np.array_equal(self.q, READY):
                    self.homing = False
                    print(f"[teleop] {self.side} rehomed", flush=True)
        if grip is not None:
            self.command[6] = grip
        self.send()


# Compile the solver before the motors are enabled: the first solve takes seconds (and can
# starve the CAN threads), every one after it a few milliseconds.
ik.solve(*ik.tcp_pose(READY), READY, CONTROL_PERIOD)
adamo_robot = connect(args)
subscribe_xr_control(adamo_robot, on_xr, max_age=0.25)
# attach_video only queues the graph; run() is what starts the Rust pipeline, and it
# blocks -- so it goes in a daemon thread and the control loop keeps the main one.
video = threading.Thread(target=adamo_robot.run, daemon=True)
video.start()
arms = [TeleopArm(adamo_robot, channel, side) for channel, side in ARMS]
print(f"[teleop] {NAME}: grip to move an arm, trigger for its gripper", flush=True)

try:
    # Ramp to the ready pose: the home pose sits on joint2/joint3's lower bound.
    starts = [arm.command[:6].copy() for arm in arms]
    rate = adamo.Rate(1 / CONTROL_PERIOD)
    for i in range(1, 201):
        for arm, q0 in zip(arms, starts, strict=True):
            arm.command[:6] = q0 + (READY - q0) * (i / 200)
            arm.robot.command_joint_pos(arm.command)
        rate.sleep()

    # Rate keeps a fixed schedule, so the milliseconds the solve costs don't stretch CONTROL_PERIOD,
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
