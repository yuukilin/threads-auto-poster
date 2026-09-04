#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
project_dir="${script_dir:h}"
helper="${project_dir}/.local/bin/keychain_update"
service="com.yuukilin.threads-auto-poster.access-token"
account="lin.yusei"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: macOS Keychain is required" >&2
  exit 1
fi

"${script_dir}/build_keychain_helper.sh"
"$helper" prompt-update "$account" "$service" "Threads API token - lin.yusei"
echo "saved: macOS Keychain service=${service}, account=${account}"
