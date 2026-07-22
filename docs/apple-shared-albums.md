# Apple Shared Albums and privacy

FrameFeed v0.1 supports an Apple Shared Album through Apple's **Public Website**
feature. This is convenient but has an important tradeoff: anyone who obtains the
public Apple URL can view the album without signing in.

FrameFeed does not ask for or store:

- your Apple ID or password;
- a two-factor authentication code;
- iCloud cookies or a trusted-session token; or
- the names of people in a photo.

The album URL is stored only in `secrets/icloud_shared_album_url`, mounted into the
worker read-only, and never copied to the generated website or logs. Downloaded
files use Apple's opaque item identifiers instead of original filenames.

## Safer operating rules

1. Use a dedicated Shared Album containing only pictures intended for the frame.
2. Do not publish its Apple URL, include it in screenshots, or commit it to Git.
3. Protect the FrameFeed page separately with an authenticated HTTPS proxy.
4. Disable **Public Website** in Photos if the link is ever exposed.
5. Rotate `secrets/frame_path` and restart both containers if the FrameFeed URL is
   exposed.

Apple can change this undocumented web interface. FrameFeed treats failures as
temporary and keeps serving the last good frame.
