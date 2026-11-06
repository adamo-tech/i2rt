import os
import signal
import threading
import time

import adamo
import jaxlie
import numpy
import tyro
from adamo.xr import Joy, PoseStamped, XrController, XrFrame, subscribe_xr_control
from yam import CONTROL_PERIOD, HOME_SPEED, READY, Args, Arm, arm_channels, connect
from yam_ik import VELOCITY_LIMITS, YamIK

TARGET_TIMEOUT = 0.25
XR2W = numpy.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

args = tyro.cli(Args)
targets = {side: None for _, side in arm_channels(args)}
hands = {
    side: {"pos": None, "rot": None, "clutch": False, "grip": None, "home": False} for _, side in arm_channels(args)
}
lock = threading.Lock()
is_stopping = threading.Event()


def on_target(side: str, sample: object) -> None:
    targets[side] = (time.monotonic(), numpy.asarray(sample.message["action"], dtype=float))


def quat_to_mat(q: object) -> numpy.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return numpy.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def on_xr(sample: object) -> None:
    msg = sample.message
    if isinstance(msg, XrFrame):
        updates = [(side, c.pose, c) for side in hands if (c := getattr(msg, side)) is not None]
    else:
        side, kind = sample.channel.split("/")[-2:]
        updates = [(side, msg if kind == "pose" and isinstance(msg, PoseStamped) else None, msg)]
    with lock:
        for name, pose, joy in updates:
            if name not in hands:
                continue
            hand = hands[name]
            if pose is not None:
                hand["pos"] = XR2W @ numpy.array([pose.position.x, pose.position.y, pose.position.z])
                hand["rot"] = XR2W @ quat_to_mat(pose.orientation) @ XR2W.T
            if not isinstance(joy, (Joy, XrController)):
                continue
            hand["clutch"] = len(joy.buttons) > 1 and bool(joy.buttons[1])
            hand["home"] = len(joy.buttons) > 5 and bool(joy.buttons[5])
            trigger = (
                float(joy.axes[4])
                if len(joy.axes) > 4 and joy.axes[4] > 0
                else float(bool(joy.buttons and joy.buttons[0]))
            )
            if trigger > 0.05:
                hand["grip"] = 1.0 - trigger


ik = YamIK()


class PolicyArm(Arm):
    def __init__(self, adamo_robot: adamo.Robot, channel: str, side: str) -> None:
        super().__init__(adamo_robot, channel, side)
        self.anchor = None
        self.is_homing = True

    def step(self, is_following_allowed: bool) -> None:
        with lock:
            hand = dict(hands[self.side])
        target = targets[self.side]
        is_tracking = is_following_allowed and hand["clutch"] and hand["pos"] is not None
        is_following = (
            is_following_allowed
            and not is_tracking
            and target is not None
            and time.monotonic() - target[0] < TARGET_TIMEOUT
        )
        if not is_tracking:
            self.anchor = None
        if hand["home"] and not is_tracking:
            self.is_homing = True

        if is_tracking:
            self.is_homing = False
            if self.anchor is None:
                position, wxyz = ik.tcp_pose(self.command[:6])
                self.anchor = (hand["pos"], hand["rot"], position, numpy.asarray(jaxlie.SO3(wxyz).as_matrix()))
            hand0, hand_rot0, tcp0, tcp_rot0 = self.anchor
            wxyz = numpy.asarray(jaxlie.SO3.from_matrix(hand["rot"] @ hand_rot0.T @ tcp_rot0).wxyz)
            self.command[:6], _, _ = ik.solve(tcp0 + (hand["pos"] - hand0), wxyz, self.command[:6], CONTROL_PERIOD)
            if hand["grip"] is not None:
                self.command[6] = hand["grip"]
        elif self.is_homing or not is_following_allowed:
            self.command[:6] += numpy.clip(
                READY - self.command[:6], -HOME_SPEED * CONTROL_PERIOD, HOME_SPEED * CONTROL_PERIOD
            )
            if numpy.array_equal(self.command[:6], READY):
                self.is_homing = False
        elif is_following:
            self.command[:6] += numpy.clip(
                target[1][:6] - self.command[:6], -VELOCITY_LIMITS * CONTROL_PERIOD, VELOCITY_LIMITS * CONTROL_PERIOD
            )
            self.command[6] = numpy.clip(target[1][6], 0.0, 1.0)

        self.send()


position, wxyz = ik.tcp_pose(READY)
ik.solve(
    position,
    numpy.asarray(jaxlie.SO3.from_matrix(numpy.asarray(jaxlie.SO3(wxyz).as_matrix())).wxyz),
    READY,
    CONTROL_PERIOD,
)
adamo_robot = connect(args)

for _, side in arm_channels(args):
    adamo_robot.on(f"{side}/joint_cmd", lambda sample, side=side: on_target(side, sample), max_age=TARGET_TIMEOUT)

subscribe_xr_control(adamo_robot, on_xr, max_age=0.25)
video_thread = threading.Thread(target=adamo_robot.run, daemon=True)
video_thread.start()
arms = [PolicyArm(adamo_robot, channel, side) for channel, side in arm_channels(args)]
signal.signal(signal.SIGINT, lambda *_: is_stopping.set())
signal.signal(signal.SIGTERM, lambda *_: is_stopping.set())
rate = adamo.Rate(1 / CONTROL_PERIOD)

try:
    while (
        video_thread.is_alive()
        and not is_stopping.is_set()
        and all(arm.robot._server_thread.is_alive() for arm in arms)
    ):
        for arm in arms:
            arm.step(is_following_allowed=True)

        rate.sleep()

    while not all(numpy.array_equal(arm.command[:6], READY) for arm in arms):
        for arm in arms:
            arm.step(is_following_allowed=False)

        rate.sleep()

    time.sleep(1.0)
finally:
    for arm in arms:
        arm.robot.close()

    adamo_robot.close()

os._exit(0)
