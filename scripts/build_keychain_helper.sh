#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
project_dir="${script_dir:h}"
source_file="${script_dir}/keychain_update.c"
target_dir="${project_dir}/.local/bin"
target_file="${target_dir}/keychain_update"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this helper requires macOS" >&2
  exit 1
fi
if [[ ! -f "$source_file" || ! -x /usr/bin/clang ]]; then
  echo "error: macOS Command Line Tools are required" >&2
  exit 1
fi

mkdir -p "$target_dir"
if [[ -x "$target_file" && "$target_file" -nt "$source_file" && "$target_file" -nt "$0" ]]; then
  echo "ready: $target_file"
  exit 0
fi

temporary="$(mktemp "${target_dir}/keychain_update.XXXXXX")"
trap 'rm -f "$temporary"' EXIT

/usr/bin/clang \
  -x objective-c -fobjc-arc -std=c11 -O2 -Wall -Wextra -Werror \
  -framework Cocoa -framework Security -framework CoreFoundation \
  "$source_file" -o "$temporary"
chmod 700 "$temporary"
mv "$temporary" "$target_file"

echo "built: $target_file"
