#!/usr/bin/env bash
set -e
cd /opt/bim2sim_workspace/bim2sim_repo
git checkout -- .
git apply /opt/bim2sim_workspace/bim2sim_patches.diff
echo "bim2sim-Patches erfolgreich eingespielt!"
