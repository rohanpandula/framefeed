# Unraid setup

This walkthrough assumes the FrameFeed project is stored at
`/mnt/user/appdata/framefeed` and your photos are in
`/mnt/user/photos/family-frame`. Change those two paths if yours differ.

## 1. Open an Unraid terminal

In the Unraid web interface, click the **Terminal** icon in the upper-right corner.

## 2. Download and initialize FrameFeed

Paste this whole block:

```bash
cd /mnt/user/appdata
git clone https://github.com/rohanpandula/framefeed.git
cd framefeed
./scripts/init.sh
sed -i 's|PHOTO_DIR_HOST=.*|PHOTO_DIR_HOST=/mnt/user/photos/family-frame|' .env
sed -i 's|PUID=.*|PUID=99|' .env
sed -i 's|PGID=.*|PGID=100|' .env
chown -R 99:100 data secrets
```

If the photo folder does not exist yet:

```bash
mkdir -p /mnt/user/photos/family-frame
```

Copy at least one photo into that folder.

## 3. Optional: connect an Apple Shared Album

In Photos on your iPhone or Mac, open the Shared Album, turn on **Public Website**,
and copy its `icloud.com/sharedalbum` link. Then run this command, replacing only
the text between the single quotes:

```bash
printf '%s\n' 'PASTE_THE_APPLE_SHARED_ALBUM_LINK_HERE' \
  > /mnt/user/appdata/framefeed/secrets/icloud_shared_album_url
```

No Apple sign-in or pairing code is required. The public-link privacy tradeoff is
explained in [Apple Shared Albums and privacy](apple-shared-albums.md).

## 4. Start it

```bash
cd /mnt/user/appdata/framefeed
docker compose up -d
```

The first build may take several minutes. Check the two containers:

```bash
docker compose ps
```

Both should say `Up`; the worker becomes `healthy` after its first successful frame.

## 5. Get the display URL

```bash
cd /mnt/user/appdata/framefeed
printf 'http://YOUR_UNRAID_IP:8080/%s/\n' "$(sed -n '1p' secrets/frame_path)"
```

Replace `YOUR_UNRAID_IP` with the address you normally use to open Unraid. Test the
result in a browser on the same network.

For a display in another home, do not expose port 8080 directly. Use an HTTPS proxy
with authentication. If you add an IP restriction, keep authentication too: home
internet addresses are not always truly static.

## Updating

```bash
cd /mnt/user/appdata/framefeed
git pull --ff-only
docker compose pull
docker compose up -d --build
```

Your `.env`, secrets, state, and photos are ignored by Git and remain in place.

## Useful checks

```bash
docker compose logs --tail=100 worker
docker compose ps
```

FrameFeed never deletes an item removed from an Apple Shared Album. It moves the
local copy under the private photo folder's `.removed` directory.
