#!/usr/bin/env bash
set -euo pipefail
: "${DRAWIO_DESKTOP_BIN:?Set DRAWIO_DESKTOP_BIN to the DrawIO Desktop executable}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DRAWIO_DISABLE_UPDATE=true
runtime_dir=$(mktemp -d /tmp/figagent-runtime.XXXXXXXX)
profile_dir=$(mktemp -d /tmp/figagent-profile.XXXXXXXX)
cleanup() { rm -rf -- "$runtime_dir" "$profile_dir"; }
trap cleanup EXIT
chmod 700 "$runtime_dir"
export XDG_RUNTIME_DIR="$runtime_dir"
xvfb-run -a dbus-run-session -- "$DRAWIO_DESKTOP_BIN" \
  --no-sandbox --disable-update --password-store=basic \
  --user-data-dir="$profile_dir" --disable-gpu --use-gl=swiftshader \
  --disable-dev-shm-usage "$@"
