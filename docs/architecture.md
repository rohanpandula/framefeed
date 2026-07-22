# Architecture

FrameFeed deliberately has no public application server or database.

## Worker

The Python worker scans a mounted folder, optionally mirrors an Apple Shared Album,
updates private state, selects a photo, and renders one exact-size JPEG. It runs as
a non-root user and does not accept inbound requests.

Private state in `data/state` includes:

- `state.json`: first-seen times, selection history, and the current frame ID;
- `image-analysis.jsonl`: normalized face boxes and layout metadata;
- `icloud-shared-manifest.json`: anonymous asset checksums and capture dates; and
- `health.json`: heartbeat and last-success status.

Writes use temporary files followed by an atomic rename where replacement matters.
The JSONL cache is append-only. A layout change can reuse face boxes without running
the detector again.

## Renderer

YuNet runs locally on a preview capped at 2048 pixels. Face boxes are normalized,
expanded by a configurable margin, and combined into a protected region. The
renderer chooses one of four outcomes:

- `cover-face-safe`: fill the display with faces protected;
- `cover-center`: fill with a center crop when no face is detected;
- `contain-aspect-fallback`: retain the complete photo when the crop is too large;
- `contain-fallback`: retain the complete photo when a face group cannot fit.

No face embeddings or identities are created.

## Static web output

The worker publishes only:

```text
data/site/<64-character-secret>/index.html
data/site/<64-character-secret>/frames/frame-<revision>.jpg
```

Nginx serves that directory without listing it and attaches no-cache, no-index, and
basic hardening headers. The secret path is a bearer capability. Put authentication
in front of it for any untrusted network.
