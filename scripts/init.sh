#!/bin/sh
set -eu

project_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

umask 077
mkdir -p data/photos data/site data/state secrets

if [ ! -f .env ]; then
    cp .env.example .env
    printf '%s\n' "Created .env"
fi

if [ ! -s secrets/frame_path ]; then
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32 > secrets/frame_path
    else
        frame_secret=$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')
        printf '%s\n' "$frame_secret" > secrets/frame_path
    fi
    printf '%s\n' "Created a private 64-character frame address"
fi

if [ ! -f secrets/icloud_shared_album_url ]; then
    : > secrets/icloud_shared_album_url
    printf '%s\n' "Created an empty optional Apple Shared Album setting"
fi

chmod 600 .env secrets/frame_path secrets/icloud_shared_album_url

if [ "$(id -u)" = "0" ]; then
    puid=$(sed -n 's/^PUID=//p' .env | head -n 1)
    pgid=$(sed -n 's/^PGID=//p' .env | head -n 1)
    chown -R "${puid:-1000}:${pgid:-1000}" data secrets
fi

chmod 0750 data/photos
chmod 0755 data/site
chmod 0700 data/state secrets

printf '\nOpen this after FrameFeed starts:\n  http://YOUR-SERVER-IP:%s/%s/\n' \
    "$(sed -n 's/^FRAMEFEED_PORT=//p' .env | head -n 1)" \
    "$(sed -n '1p' secrets/frame_path)"
