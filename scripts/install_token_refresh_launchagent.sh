#!/bin/zsh
set -euo pipefail

label="com.yuukilin.threads-token-refresh"
script_dir="${0:A:h}"
project_dir="${script_dir:h}"
source_plist="${project_dir}/launchd/${label}.plist"
target_dir="${HOME}/Library/LaunchAgents"
target_plist="${target_dir}/${label}.plist"
log_dir="${HOME}/.codex/automations/threads-token-refresh"
domain="gui/$(id -u)"
python_bin="$(command -v python3)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this installer requires macOS" >&2
  exit 1
fi

if [[ ! -f "$source_plist" || ! -f "${project_dir}/threads_api.py" ]]; then
  echo "error: incomplete repository checkout: ${project_dir}" >&2
  exit 1
fi

"${script_dir}/build_keychain_helper.sh"
mkdir -p "$target_dir" "$log_dir"
temp_plist="$(mktemp "${TMPDIR:-/tmp}/${label}.XXXXXX")"
trap 'rm -f "$temp_plist"' EXIT
sed \
  -e "s|__PYTHON__|${python_bin}|g" \
  -e "s|__PROJECT_DIR__|${project_dir}|g" \
  -e "s|__LOG_DIR__|${log_dir}|g" \
  "$source_plist" > "$temp_plist"
/usr/bin/plutil -lint "$temp_plist"
cp "$temp_plist" "$target_plist"
chmod 600 "$target_plist"
/usr/bin/plutil -lint "$target_plist"

/bin/launchctl bootout "$domain/$label" 2>/dev/null || true
/bin/launchctl bootstrap "$domain" "$target_plist"
/bin/launchctl enable "$domain/$label"

echo "installed: $label"
echo "project: ${project_dir}"
echo "logs: ${log_dir}"
