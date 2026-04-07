from __future__ import annotations

"""
Bootstrap the on-disk state layout.

Usage:
  python3 -m storage.bootstrap_state

Env:
  STATE_DIR=/path/to/state  (optional; default: <repo>/state)
"""

from storage.config_io import DEFAULT_CONFIG, config_path, ensure_state_layout, save_config
from storage.registry_io import ensure_registry_layout


def main() -> None:
    ensure_state_layout()
    ensure_registry_layout()

    # Write default config.yaml only if missing
    if not config_path().exists():
        save_config(dict(DEFAULT_CONFIG))

    print(f"State initialized at: {config_path().parent}")
    print(f"- config: {config_path()}")


if __name__ == "__main__":
    main()

