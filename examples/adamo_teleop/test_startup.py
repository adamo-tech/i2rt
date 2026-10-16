"""Exercise teleop startup and fault handling without opening a CAN device."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import adamo_teleop as teleop
import numpy as np
import pytest


@pytest.fixture
def setup_runtime(monkeypatch: pytest.MonkeyPatch) -> tuple:
    events = []
    robots = []
    solver = Mock()
    solver.tcp_pose.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    solver.solve.side_effect = lambda *args: events.append("warm")
    monkeypatch.setattr(teleop, "YamIK", lambda: solver)

    def make_robot(**kwargs: object) -> Mock:
        side = str(kwargs["channel"])
        events.append(side)
        robot = Mock()
        robot.get_joint_pos.return_value = np.zeros(7)
        robot.motor_chain.running = True
        robot.close.side_effect = lambda: events.append("close " + side)
        robots.append(robot)
        return robot

    monkeypatch.setattr(teleop, "get_yam_robot", make_robot)
    network = Mock()
    network.close.side_effect = lambda: events.append("close network")
    monkeypatch.setattr(teleop.adamo, "Robot", lambda **kwargs: network)
    monkeypatch.setattr(teleop.adamo, "Recording", Mock())
    monkeypatch.setattr(teleop, "subscribe_xr_control", Mock())
    monkeypatch.setattr(teleop, "RuntimeThread", Mock())
    monkeypatch.setenv("ADAMO_API_KEY", "test-only")

    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(teleop.time, "sleep", interrupt)
    return events, robots, network


def test_solver_warms_before_motors_and_shutdown_closes_both(setup_runtime: tuple) -> None:
    events, robots, _ = setup_runtime
    with pytest.raises(KeyboardInterrupt):
        teleop.main(teleop.Args(arm_left="left", arm_right="right"))
    assert events == ["warm", "left", "right", "close right", "close left", "close network"]
    assert all(robot.command_joint_pos.called for robot in robots)


def test_network_close_failure_still_closes_both_arms(setup_runtime: tuple) -> None:
    events, _, network = setup_runtime
    network.close.side_effect = ConnectionError("disconnected")
    with pytest.raises(ConnectionError, match="disconnected"):
        teleop.main(teleop.Args(arm_left="left", arm_right="right"))
    assert events[-2:] == ["close right", "close left"]


def test_arm_close_failure_still_closes_other_arm_and_network(setup_runtime: tuple) -> None:
    events, robots, network = setup_runtime

    def fail_step(_: object) -> None:
        robots[1].close.side_effect = RuntimeError("close failed")
        raise KeyboardInterrupt

    # Fail only once both arms have been constructed.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(teleop.time, "sleep", fail_step)
        with pytest.raises(RuntimeError, match="close failed"):
            teleop.main(teleop.Args(arm_left="left", arm_right="right"))
    assert "close left" in events
    network.close.assert_called_once()


@pytest.mark.parametrize("failure", ["chain", "motor_thread", "robot_thread"])
def test_motor_failure_stops_before_next_command(failure: str) -> None:
    robot = Mock()
    robot.motor_chain.running = True
    if failure == "chain":
        robot.motor_chain.running = False
    elif failure == "motor_thread":
        robot.motor_chain._control_thread.is_alive.return_value = False
    else:
        robot._server_thread.is_alive.return_value = False
    with pytest.raises(RuntimeError, match=r"left .*control loop stopped"):
        teleop.check_health([SimpleNamespace(side="left", robot=robot)], Mock())
    robot.command_joint_pos.assert_not_called()


def test_runtime_exception_reaches_control_loop() -> None:
    def fail() -> None:
        raise ConnectionError("transport closed")

    runtime = teleop.RuntimeThread(fail)
    runtime.start()
    runtime.join(timeout=2)
    with pytest.raises(RuntimeError, match="Adamo runtime failed") as exc:
        runtime.check()
    assert isinstance(exc.value.__cause__, ConnectionError)


def test_runtime_unexpected_return_reaches_control_loop() -> None:
    runtime = teleop.RuntimeThread(lambda: None)
    runtime.start()
    runtime.join(timeout=2)
    with pytest.raises(RuntimeError, match="Adamo runtime stopped"):
        runtime.check()


def test_live_runtime_is_healthy() -> None:
    stop = threading.Event()
    runtime = teleop.RuntimeThread(stop.wait)
    runtime.start()
    try:
        runtime.check()
    finally:
        stop.set()
        runtime.join(timeout=2)
