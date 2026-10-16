import os
import signal
import threading
import time

import adamo
import numpy
import tyro
from yam import CONTROL_PERIOD, HOME_SPEED, READY, Args, Arm, arm_channels, connect
from yam_ik import VELOCITY_LIMITS

TARGET_TIMEOUT = 0.25

args = tyro.cli(Args)
targets = {side: None for _, side in arm_channels(args)}
is_stopping = threading.Event()


def on_target(side: str, sample: object) -> None:
    targets[side] = (time.monotonic(), numpy.asarray(sample.message["action"], dtype=float))


class PolicyArm(Arm):
    def step(self, is_following_allowed: bool) -> None:
        target = targets[self.side]

        if is_following_allowed and target is not None and time.monotonic() - target[0] < TARGET_TIMEOUT:
            self.command[:6] += numpy.clip(target[1][:6] - self.command[:6], -VELOCITY_LIMITS * CONTROL_PERIOD, VELOCITY_LIMITS * CONTROL_PERIOD)
            self.command[6] = numpy.clip(target[1][6], 0.0, 1.0)
        else:
            self.command[:6] += numpy.clip(READY - self.command[:6], -HOME_SPEED * CONTROL_PERIOD, HOME_SPEED * CONTROL_PERIOD)

        self.send()


adamo_robot = connect(args)

for _, side in arm_channels(args):
    adamo_robot.on(f"{side}/joint_cmd", lambda sample, side=side: on_target(side, sample), max_age=TARGET_TIMEOUT)

video_thread = threading.Thread(target=adamo_robot.run, daemon=True)
video_thread.start()
arms = [PolicyArm(adamo_robot, channel, side) for channel, side in arm_channels(args)]
signal.signal(signal.SIGINT, lambda *_: is_stopping.set())
signal.signal(signal.SIGTERM, lambda *_: is_stopping.set())
rate = adamo.Rate(1 / CONTROL_PERIOD)

try:
    while video_thread.is_alive() and not is_stopping.is_set() and all(arm.robot._server_thread.is_alive() for arm in arms):
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
