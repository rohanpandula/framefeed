# reTerminal E1004 setup

FrameFeed pre-renders the exact pixels the display needs. The reTerminal should
load the live page, rather than a screenshot made by the SenseCraft editor.

## Landscape

Use these values in `.env`:

```dotenv
FRAME_WIDTH=1600
FRAME_HEIGHT=1200
MIN_CROP_RETAINED_FRACTION=0.75
```

In SenseCraft HMI:

1. Add a **Web** widget to the canvas.
2. Choose **Live iframe** for **Render mode**.
3. Enter preview width `1600` and preview height `1200`.
4. Paste the private FrameFeed HTTPS URL.
5. Stretch the widget to fill the canvas, save, and deploy it to the device.

## Portrait

Use `FRAME_WIDTH=1200` and `FRAME_HEIGHT=1600`, restart the worker, and use those
same numbers as the preview width and height.

```bash
docker compose restart worker
```

## Why Live iframe?

**Preview image** captures a fixed picture while editing. **Live iframe** lets the
device request FrameFeed again on its normal refresh cycle, so new photos appear
without your parents touching anything.

## Why photos are not stretched

FrameFeed always preserves the source aspect ratio. It first attempts a crop that
keeps detected faces plus a safety margin. If the required crop would discard more
than 25% of the image—or cannot safely contain a group of faces—it shows the whole
photo over a dim, blurred copy that fills the unused area.

If a network fetch temporarily fails, the server keeps the last successfully
rendered frame rather than publishing a broken page.
