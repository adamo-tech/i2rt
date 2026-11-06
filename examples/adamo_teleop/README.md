# YAM teleop with stereo and wrist video

The operator's view is two USB cameras, one per eye, each streamed as its own
track (`zed_left` / `zed_right`, marked `stereo_left` / `stereo_right`); the
headset pairs them by capture time into one stereo view. Each eye captures
2048x1536 MJPEG at 30 fps, which GStreamer decodes on the GPU (`vajpegdec`,
`vapostproc`) into VA memory for Adamo's native VA-API encoder, at a 4000 kbit/s
VBR target with adaptive bitrate. No image buffers cross into Python.

Each arm that runs also streams its wrist-mounted RealSense D405's color image
(`wrist_left` / `wrist_right`) as 1280x720 YUYV, converted to NV12 on the CPU,
at 2000 kbit/s.

Every track uses a 10-second keyframe distance: the VA-API encoder has no intra
refresh, and at the 2-second default each IDR frame spiked latency past 300 ms.

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
