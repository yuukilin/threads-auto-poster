#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
project_dir="${script_dir:h}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this setup requires macOS" >&2
  exit 1
fi

cd "$project_dir"

"$script_dir/build_keychain_helper.sh"
python3 -m unittest discover -s tests -v
python3 -m py_compile threads_api.py

if ! python3 threads_api.py verify; then
  echo "Threads 權杖尚未設定或已失效，現在開啟安全輸入框。"
  "$script_dir/configure_token_in_terminal.sh"
  python3 threads_api.py verify
fi

"$script_dir/install_token_refresh_launchagent.sh"

echo "Mac-side setup complete."
echo "Next: create or reconcile the Codex schedule from the shared automation template."
