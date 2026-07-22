# Security policy

## Supported versions

Security fixes are provided for the latest release. FrameFeed is currently alpha
software and should be deployed behind a trusted LAN, VPN, or authenticated proxy.

## Reporting a vulnerability

Please use GitHub's private **Report a vulnerability** flow on the Security tab. Do
not open a public issue containing exploit details, album links, secrets, addresses,
or private photos. You should receive an acknowledgment within seven days.

## Threat model

FrameFeed minimizes its public surface: the worker has no listening port, and Nginx
serves only generated static files under a high-entropy path. The path protects
against casual discovery but is still a bearer secret. It does not defend against a
leaked URL, a compromised host, malicious source images, or an exposed Apple public
album link.

For internet access, use HTTPS and authentication. An IP allow-list is defense in
depth, not primary authentication. Never publish the photo, state, or secrets
mounts. Keep Docker, the host OS, and the reverse proxy updated.
