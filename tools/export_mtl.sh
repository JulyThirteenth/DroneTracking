#!/usr/bin/env bash

SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scenes/behavior1k/export_mtl.sh"
unset SCRIPT_DIR
