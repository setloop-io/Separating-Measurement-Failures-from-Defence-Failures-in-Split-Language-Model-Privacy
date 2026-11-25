#!/usr/bin/env bash
# Phase-2 CPU node env setup: venv + torch CPU + package + gpt2 warm cache.
# Runs detached via nohup; log at ~/setup_env.log.
set -euo pipefail
export HF_HUB_DISABLE_XET=1
cd "$HOME/llm-attacker"
exec > "$HOME/setup_env.log" 2>&1
set -x

[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -e . transformers numpy scipy
.venv/bin/python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('gpt2'); AutoTokenizer.from_pretrained('gpt2'); print('env OK')"
echo SETUP_DONE
