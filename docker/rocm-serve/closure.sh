#!/bin/sh
# closure.sh OUT LIB... : copies each LIB and the /opt part of its ldd closure under OUT, keeping paths and the soname
# symlink chains (RPATH $ORIGIN lookups still resolve). System libraries (/lib64) come from the runtime image's dnf.
set -eu
out=$1; shift
for l in "$@"; do echo "$l"; ldd "$l" | awk '$3 ~ /^\/opt\// {print $3}'; done | sort -u | while read -r p; do
    cp -a --parents "$p" "$out"
    while [ -L "$p" ]; do
        t=$(readlink "$p"); case $t in /*) p=$t ;; *) p=$(dirname "$p")/$t ;; esac
        cp -a --parents "$p" "$out"
    done
done
