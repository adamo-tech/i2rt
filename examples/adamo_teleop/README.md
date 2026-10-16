# YAM teleop with ZED stereo video

The ZED captures 1920x1080 per eye as one 3840x1080 YUYV frame. The
GStreamer pipeline filters corrupt buffers, bounds the raw queue to two frames,
converts YUY2 to NV12, and uses `videorate` to repeat intact frames over capture
gaps at a nominal 30 FPS. Adamo's native Intel VA-API backend encodes H.264 with
an 8000 kbit/s VBR target with network bitrate adaptation disabled. Simple
scenes may use fewer bits; CBR padding is not enabled. The installed SDK
uses a half-second coded-bit reservoir for keyframe quality, without
queuing raw frames. No image buffers cross into Python.

The camera's USB corruption is still present at this resolution. Output FPS
includes repeats, not newly captured motion: a 20-second diagnostic captured
593 buffers, rejected 61 flagged corrupt, and produced 598 output frames with
67 repeats (including startup timing adjustment). `videorate` fills gaps when
another frame arrives, so it does not guarantee uninterrupted wall-clock pacing
during a camera stall. This avoids restarting capture on each bad USB frame.
Use 720p per eye when fresh-frame consistency matters more than resolution.

`zedsrc` is not used: this host has Intel graphics, while that plugin depends
on the NVIDIA/CUDA-based ZED SDK. The installed `v4l2src`, `identity`, `queue`,
`videoconvert`, `videorate`, and `capsfilter` elements are sufficient.

## Service on the London host

`yam-teleop.service` runs as `anvil` from `/home/anvil/i2rt`, using the existing
uv environment without synchronizing or upgrading dependencies at startup.
The API key is read from `/home/anvil/adamo.env` (mode 0600), which contains
`ADAMO_API_KEY=...`. Never put the key in this repository or the unit file.
The stable ZED capture path survives `/dev/videoN` renumbering.
Both `can_left` and `can_right` must be configured and up at 1 Mbit/s.

Starting the service initializes both arms and ramps them to the ready pose.
Clear the workspace before starting it. Enabling the service also starts this
sequence on subsequent boots. Automatic crash restarts are disabled because
restarting would move the arms again; diagnose the failure before restarting.

From the repository root, after installing the project dependencies:

```sh
sudo install -m 0644 examples/adamo_teleop/yam-teleop.service /etc/systemd/system/yam-teleop.service
sudo systemctl daemon-reload
sudo systemctl enable yam-teleop.service
sudo systemctl start yam-teleop.service
```

Check or stop it with:

```sh
systemctl status yam-teleop.service --no-pager
journalctl -u yam-teleop.service -f
sudo systemctl stop yam-teleop.service
```

Stop the service before running the example manually or accessing the camera
with another program. Service shutdown sends SIGINT so the example closes the
Adamo session and arm connections. Edit the user, absolute paths, CAN names,
and camera path in the unit when installing on another host. After changing
the unit, reinstall it and run `systemctl daemon-reload`; restart only when the
arm workspace is clear.

## Startup and fault handling

The example compiles the IK solver before opening either arm. Cold JAX
compilation previously ran with the CAN loops active and coincided with motor
communication faults; a hardware-free timing check measured a 76 ms Python
thread pause during compilation versus 1.2 ms after warmup.

The main loop checks both motor-control threads, both robot-server threads,
and the Adamo runtime during the ready-pose ramp and every teleop tick. A
thread failure ends the service. Arm connections close before the network
session, and cleanup continues if an individual close raises. Automatic
restarts remain disabled.

Run the hardware-free regression checks with:

```sh
uv run --no-sync pytest examples/adamo_teleop/test_startup.py -q
```
