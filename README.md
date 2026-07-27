# FrameFeed

[![CI](https://github.com/rohanpandula/framefeed/actions/workflows/ci.yml/badge.svg)](https://github.com/rohanpandula/framefeed/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/rohanpandula/framefeed)](https://github.com/rohanpandula/framefeed/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**A privacy-first photo pipeline for e-paper and other slow displays.**

FrameFeed turns a local photo folder—or an optional Apple Shared Album—into one
carefully composed, static image page. It favors recent photos without forgetting
old ones, keeps faces in frame, avoids distorted aspect ratios, and preserves the
last good image when a source is unavailable.

> **v0.1 is an early release.** Back up your configuration, review the security
> notes, and expect a few rough edges.

## Why I made it

I bought a Seeed Studio reTerminal E1004 for my parents, who are in their
seventies, so new photos of their granddaughter could simply appear at home.
They should not need another app, account, or set of instructions just to enjoy
family pictures. FrameFeed grew from that small, personal goal.

## Features

- Mirrors a NAS folder or an experimental Apple Shared Album public website.
- Offers Basic face-only CV cropping with the compact, MIT-licensed OpenCV
  YuNet model and an optional Advanced face-and-person pipeline.
- Crops toward the primary people without stretching photos; falls back to the
  complete image over a blurred background when a safe crop is uncertain.
- Shows a newly discovered photo for 65 minutes, then returns to a deterministic,
  recency-weighted shuffle with a 24-photo no-repeat window.
- Writes face boxes, layout decisions, and descriptions to a private JSONL cache.
- Publishes only an anonymous rendered JPEG and tiny HTML page at a secret path.
- Keeps the last successful frame online during source or detector outages.
- Includes landscape and portrait presets for the reTerminal E1004.
- Runs as non-root containers with read-only web serving and no directory listing.

## How it works

```mermaid
flowchart LR
    A["NAS folder"] --> C["Private photo catalog"]
    B["Apple Shared Album"] --> C
    C --> D["Recent-biased selector"]
    D --> E["Basic or Advanced CV renderer"]
    E --> F["One anonymous JPEG + HTML"]
    F --> G["E-paper display"]
```

Original images and analysis metadata stay on your server. The display receives
only the current pre-rendered frame.

## CV cropping modes

FrameFeed has two useful levels of computer-vision cropping:

| Mode | Detectors | Hardware | Best for |
| --- | --- | --- | --- |
| **Basic** | OpenCV YuNet faces | CPU; no GPU required | Simple installs and modest photo libraries |
| **Advanced** | SCRFD `buffalo_l` faces plus YOLO26s people | NVIDIA CUDA, Intel OpenVINO, or CPU | Small or obscured subjects, groups, and busy backgrounds |

### Basic: face-aware cropping

Basic mode is the original FrameFeed cropper and remains the default. It finds
faces, protects configurable space around them, and shifts the largest
target-aspect crop that keeps the group safe. If the photo is too tall, the
group is too wide, or detection is unavailable, it displays the complete image
over a blurred fill.

Basic mode is deliberately lightweight. It runs on ordinary x86-64 and ARM64
CPUs and does not require CUDA, OpenVINO, or a separate inference service.

### Advanced: subject-aware cropping

Advanced mode adds foreground-person detection to the face boxes. Its cropper:

- merges duplicate person detections;
- ignores tiny, distant background people;
- keeps faces and useful upper-body context instead of requiring every foot;
- caps face-only zoom when another undetected person may be present; and
- returns to the Basic decision whenever a tighter crop cannot be proven safe.

The reference deployment uses Immich's SCRFD `buffalo_l` face detector and a
YOLO26s person detector. Models and embeddings are not published with the
rendered frame; only normalized boxes and the crop decision are retained in
private state. The Advanced detector service and corpus evaluation utilities
are in [`scripts/person_service.py`](scripts/person_service.py),
[`scripts/detect_people.py`](scripts/detect_people.py), and
[`scripts/evaluate_crops.py`](scripts/evaluate_crops.py). Advanced mode is
currently an opt-in deployment profile rather than the default Compose stack.

#### Advanced hardware backends

- **NVIDIA GPU:** run the detector through CUDA. The reference Unraid deployment
  uses an RTX A4000.
- **Intel integrated GPU:** [export the YOLO model to OpenVINO](https://docs.ultralytics.com/modes/export/),
  pass
  `/dev/dri` into the detector container, and compile the model for `GPU`.
  OpenVINO reports the integrated GPU as `GPU.0` (or simply `GPU`).
- **CPU:** OpenVINO can run the exported model on `CPU`, but analysis will take
  longer.

An Intel iGPU does not use the CUDA container or VAAPI/Quick Sync inference
path. It needs the OpenVINO runtime and Intel compute driver. Because FrameFeed
analyzes still photos in the background rather than real-time video, an Intel
iGPU can be a practical way to reserve a discrete GPU for other workloads.
Benchmark the complete corpus before changing a working deployment. See
OpenVINO's [Intel GPU configuration guide](https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html)
for driver and container requirements. The Advanced model and Ultralytics
runtime are not bundled; review their upstream licenses before redistribution.

On the reference server, an Intel UHD 770 processed a 340-photo corpus in about
72 seconds and reproduced 99.1% of the A4000 square-input crop decisions within
a 2% coordinate tolerance. It is fast enough for FrameFeed's background
analysis, although the A4000 remains roughly 3.3× faster under matched
preprocessing. See the [Advanced CV backend benchmark](docs/advanced-cv-benchmark.md)
for the method and complete results.

## Prerequisites

- Docker Engine with Docker Compose v2; Unraid is supported.
- A folder of JPEG, PNG, HEIC, HEIF, or WebP images.
- About 1 GB of free space for the container image and local state.
- An HTTPS reverse proxy, VPN, or private LAN if the display is outside your home.
- Advanced mode additionally needs a supported inference backend and more
  container storage; Basic mode has no GPU requirement.

## Installation

Clone the project and create the private settings:

```bash
git clone https://github.com/rohanpandula/framefeed.git
cd framefeed
./scripts/init.sh
```

Open `.env` and set `PHOTO_DIR_HOST` to your photo folder. For Unraid, it may look
like this:

```dotenv
PHOTO_DIR_HOST=/mnt/user/photos/family-frame
```

Start FrameFeed:

```bash
docker compose up -d
```

The setup script prints the private frame URL. To print it again:

```bash
printf 'http://YOUR-SERVER-IP:8080/%s/\n' "$(sed -n '1p' secrets/frame_path)"
```

Put a test photo in your source folder, wait up to five minutes, and open that URL.
For a copy-and-paste Unraid walkthrough, see [Unraid setup](docs/unraid.md).

## Apple Shared Album source

FrameFeed does **not** need your Apple ID, password, 2FA code, or `icloudpd`. It can
read a Shared Album only after you deliberately enable Apple's **Public Website**
option. Paste that public URL into the local secret file:

```bash
printf '%s\n' 'https://www.icloud.com/sharedalbum/#YOUR_ALBUM_ID' \
  > secrets/icloud_shared_album_url
docker compose restart worker
```

Anyone who obtains the Apple public-album URL can view that album. Read
[Apple Shared Albums and privacy](docs/apple-shared-albums.md) before enabling it.

## Display setup

For a landscape reTerminal E1004:

1. Keep `FRAME_WIDTH=1600`, `FRAME_HEIGHT=1200`, and `FRAME_ROTATION=0` in `.env`.
2. In SenseCraft HMI, add a **Web** widget.
3. Choose **Live iframe** as the render mode.
4. Set preview width to `1600` and preview height to `1200`.
5. Paste the private HTTPS frame URL and publish it to the device.

If the device is rotated, copy the values from
[`presets/reterminal-e1004-portrait.env`](presets/reterminal-e1004-portrait.env).
More detail is in [reTerminal E1004 setup](docs/reterminal-e1004.md).

## Selection logic

FrameFeed makes the rotation feel shuffled while giving recent additions more
chances to appear:

```text
weight(age in days) = 1 + 6 × e^(-age / 21)
```

A photo added today has weight 7. The extra bias decays over roughly three weeks;
months-old photos approach the same baseline weight. The choice is stable within
each rotation window, so restarts do not unexpectedly change the frame.

## Configuration

| Setting | Default | Purpose |
| --- | ---: | --- |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `1600` / `1200` | Rendered frame size |
| `FRAME_ROTATION` | `0` | Clockwise output rotation (`0`, `90`, `180`, or `270`) |
| `ROTATION_SECONDS` | `3600` | Normal display time |
| `NEW_PHOTO_HOLD_SECONDS` | `3900` | Time reserved for a new photo |
| `SCAN_INTERVAL_SECONDS` | `300` | Folder scan frequency |
| `MIN_CROP_RETAINED_FRACTION` | `0.75` | Use contain mode below this safe-crop fraction |
| `MIN_FACE_CROP_RETAINED_FRACTION` | `0.50` | Allow tighter crops when detected faces remain safe |
| `FACE_DETECTOR` | `yunet` | `yunet`, `immich`, or `none` |
| `FACE_MARGIN_RATIO` | `0.75` | Space protected around detected faces |
| `ANALYSIS_BATCH_SIZE` | `8` | Background analyses per scan |
| `PUID` / `PGID` | `1000` / `1000` | Host ownership used by the worker (`99` / `100` on Unraid) |

All options and comments are in [`.env.example`](.env.example).

## Security

The generated path is deliberately difficult to guess, and FrameFeed removes
source filenames and disables indexing. Treat the path like a password: anybody
who has it can see the current picture.

For access beyond a trusted LAN, put FrameFeed behind Cloudflare Access, Tailscale,
or another authenticated reverse proxy. An IP allow-list is useful as an extra
layer, but residential IPs can change and should not be the only control. Never
port-forward the worker container or expose `data/state`, `secrets`, or the source
photo folder. See [SECURITY.md](SECURITY.md) for the threat model and reporting.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
pytest
```

Run one local update with environment variables pointing at writable test folders:

```bash
framefeed --once
```

The architecture and state files are documented in [docs/architecture.md](docs/architecture.md).

## Contributing

Bug reports and focused pull requests are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) and avoid attaching private family photos,
album URLs, frame secrets, or server addresses to issues.

## Roadmap

- Authenticated photo sources such as Immich and generic WebDAV.
- A small local setup/status page that never handles the display traffic.
- Multiple displays and independent playlists.
- Optional scene labels without identity recognition.

## License

FrameFeed is licensed under the [Apache License 2.0](LICENSE). The bundled YuNet
model is MIT-licensed; its attribution is in [NOTICE](NOTICE) and
[`models/YUNET_LICENSE`](models/YUNET_LICENSE).

FrameFeed is an independent project and is not affiliated with Apple, Seeed Studio,
OpenCV, Cloudflare, or Immich.
