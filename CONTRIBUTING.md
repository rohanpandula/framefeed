# Contributing to FrameFeed

Thanks for helping make family photo displays simpler and safer.

## Before opening an issue

- Search existing issues and the documentation.
- Remove family photos, names, Apple album links, frame secrets, domains, and IPs.
- Include the FrameFeed version, platform, container status, and sanitized logs.

## Pull requests

1. Fork the repository and create a focused branch.
2. Install the development dependencies with `pip install -e '.[dev]'`.
3. Add or update tests for behavior changes.
4. Run `ruff check .` and `pytest`.
5. Update documentation and `CHANGELOG.md` when users will notice the change.

Small, reviewable pull requests are preferred. By submitting a contribution, you
agree that it is licensed under Apache-2.0.
