#!/bin/sh
# Double-click (or: sh start-webui.sh) to start TheBrain WebUI.
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
  .venv/bin/python main.py --webui
else
  python3 main.py --webui
fi
