#!/usr/bin/env bash
# start.sh MODEL_DIR : checks the host (a gfx1151 GPU, the GTT limit, free memory), writes .env and starts the
# compose service, then waits for /health. Re-run safe; MODEL_DIR may be left out once .env exists.
# FORCE=1 skips the memory checks (not the GPU check).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dc=(docker compose -f "$here/compose.yaml")
die() { echo "start.sh: $*" >&2; exit 1; }

# gfx1151 = KFD gfx_target_version 110501 (Strix Halo's Radeon 8060S)
grep -qs '^gfx_target_version 110501$' /sys/class/kfd/kfd/topology/nodes/*/properties \
    || die "no gfx1151 GPU in /sys/class/kfd (amdgpu loaded? a Strix Halo?)"

# GTT: the GPU's memory is system RAM up to ttm.pages_limit (4 KiB pages); weights + 8 x 262k fp8 KV need ~100 GiB.
# Raise it on the kernel command line, e.g. ttm.pages_limit=27262976 (104 GiB) or 31457280 (120 GiB) and
# amd_iommu=off (GRUB_CMDLINE_LINUX_DEFAULT + update-grub, or grubby), reboot; BIOS UMA/VRAM carve-out at its minimum.
gib() { awk -v b="$1" 'BEGIN{printf "%d", b/2^30}'; }
pages=$(cat /sys/module/ttm/parameters/pages_limit 2>/dev/null || echo 0)
gtt=$(cat /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null | sort -n | tail -1)
avail=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
echo "GTT limit $(gib "${gtt:-0}") GiB (ttm.pages_limit=$pages), MemAvailable $(gib "$avail") GiB"
if [ "${FORCE:-0}" != 1 ]; then
    (($(gib "${gtt:-0}") >= 100)) || die "GTT limit under 100 GiB: set ttm.pages_limit=27262976 (see above), or FORCE=1"
    (($(gib "$avail") >= 90)) || die "MemAvailable under 90 GiB: stop other GPU/memory users first, or FORCE=1"
fi

# .env: the model dir and the host's GIDs owning /dev/kfd and /dev/dri (they vary between distros)
env=$here/.env
if [ $# -ge 1 ]; then
    [ -f "$1/config.json" ] || die "$1: no config.json (fetch-model.sh $1 first)"
    model=$(realpath "$1")
    gid() { getent group "$1" | cut -d: -f3; }
    vg=$(gid video); rg=$(gid render)
    [ -n "$vg" ] && [ -n "$rg" ] || die "no video/render group on this host"
    printf 'MODEL_DIR=%s\nVIDEO_GID=%s\nRENDER_GID=%s\n' "$model" "$vg" "$rg" > "$env"
fi
[ -f "$env" ] || die "usage: $0 MODEL_DIR"
docker image inspect tensorfold-rocm:serve >/dev/null 2>&1 \
    || die "image tensorfold-rocm:serve missing: see docker/rocm-serve/Dockerfile (docker build ... -t tensorfold-rocm:serve .)"

"${dc[@]}" up -d
echo "loading (~1-2 min from NVMe; up to 10 min from a cold page cache)..."
for _ in $(seq 600); do
    curl -fsS -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo "ready: http://127.0.0.1:8080/v1"; exit 0; }
    [ "$(docker inspect -f '{{.State.Running}}' tensorfold 2>/dev/null)" = true ] \
        || { docker logs --tail 40 tensorfold >&2; die "the server exited while loading"; }
    sleep 2
done
die "not healthy after 20 min: docker logs tensorfold"
