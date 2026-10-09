# TensorFold ROCm/HIP serving image for Qwen3.8 Flash Next on Strix Halo (gfx1151).
# Build from the repo root (needs tf-rocm-dev:latest from docker/rocm-dev; no GPU needed):
#   docker build -f docker/rocm-serve/Dockerfile -t mimiron/tensorfold:rocm-$(git rev-parse --short HEAD) .
# Stages: build (dev image: zig + hipcc + ROCm torch/Triton), rt (the HIP runtime libs' closures), final (Fedora 44).
ARG DEV_IMAGE=tf-rocm-dev:latest
ARG ROCM_IMAGE=local/ciru-toolchain:rocm10.0@sha256:0d5b4c3e850f804165b79e08b58bb239e2a33660b56b215d3a35a7126d003b4a
# the retained-PM4 runtime (pwilkin/rocm-systems ilintar-experiments 7dda3ac6cf) and the ROCm core-10.1 libs it links;
# a tag (the local copy, image ec5aa9efe595: ghcr refuses anonymous pulls), pinned by the libs' sha256 in stage rt
ARG PM4_IMAGE=ghcr.io/halo-box/qwen3.8-flash-next-strix-halo-llama:rocm-d94fa6bacb0e

FROM ${PM4_IMAGE} AS pm4
FROM ${ROCM_IMAGE} AS rocm

FROM ${DEV_IMAGE} AS build
ARG JOBS=8
USER root
COPY . /tensorfold
WORKDIR /tensorfold
# engine + HIP code objects (hipcc missing qmm_group, prefill_attention, ... is expected: docs/rocm/SERVE.md)
RUN nice -n 19 zig build -j${JOBS} -Dgpu=hip -Dhipcc=/opt/rocm/bin/hipcc --prefix /opt/tensorfold native install \
    && test -x /opt/tensorfold/native/bin/tensorfold-native
# the Triton AOT set, both specs, where tensorfold-native looks beside its binary
RUN TRITON_CACHE_DIR=/tmp/triton PYTHONPATH=/tensorfold/src nice -n 19 python -B tools/zig/flashnext_aot.py build \
      --target hip --spec zig/tests/cuda/flashnext/kernels.json --spec zig/tests/cuda/flashnext/kernels_int4ar.json \
      --jobs ${JOBS} --out /opt/tensorfold/native/share/tensorfold/cuda/gfx1151 | tee /tmp/aot.log \
    && grep -q 'kernels ->' /tmp/aot.log && ! grep -q PROBLEM /tmp/aot.log
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold \
    && cp -a LICENSE LICENSES NOTICE THIRD_PARTY_NOTICES.md /opt/tensorfold/share/doc/tensorfold/

# Each runtime's ldd closure under /opt, paths and soname links kept (RPATH $ORIGIN lookups still resolve).
FROM rocm AS rt
COPY --from=pm4 /opt/strix /opt/strix
COPY --from=pm4 /opt/rocm/core-10.1/lib /opt/rocm/core-10.1/lib
RUN cd /opt/strix/lib && printf '%s  %s\n' \
      4f4375df2743d7891a56d265f1ae760ee5ef501c93c58633ae9c1402f51beeb8 libamdhip64.so.7.16.26366-7dda3ac6cf \
      7f1ef8ed501fee30973f6266998e3834ea15452092f5aff80163575114c4bb28 libhiprtc.so.7.16.26366-7dda3ac6cf \
      016515b20f855eb38bf3197d7bf3199235734d3f279b4f60bd5d67714545e546 libhsa-runtime64.so.1.21.0 | sha256sum -c -
RUN closure() { for l in "$@"; do echo "$l"; LD_LIBRARY_PATH=$LDP ldd "$l" | awk '$3 ~ /^\/opt\// {print $3}'; done \
         | sort -u | while read -r p; do cp -a --parents "$p" /out; \
             while [ -L "$p" ]; do p=$(dirname "$p")/$(readlink "$p"); cp -a --parents "$p" /out; done; done; } \
    && mkdir -p /out/opt/doc \
    && LDP= closure /opt/rocm/core-10.0/lib/libamdhip64.so.7 \
    && LDP=/opt/strix/lib:/opt/rocm/core-10.1/lib closure /opt/strix/lib/libamdhip64.so.7 \
    && cp -a /opt/rocm/lib /opt/rocm/core /out/opt/rocm/ \
    && cp -a /opt/rocm/core-10.0/share/doc /out/opt/doc/rocm-10.0 \
    && cp -a /opt/strix/versions.txt /out/opt/doc/pm4-versions.txt

FROM registry.fedoraproject.org/fedora:44
RUN dnf -y --nodocs --setopt=install_weak_deps=False install elfutils-libelf libdrm numactl-libs libstdc++ \
    && dnf clean all && rm -rf /var/cache/dnf/*
COPY --from=rt /out/opt /opt
COPY --from=build /opt/tensorfold /opt/tensorfold
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native \
    && mv /opt/doc/* /opt/tensorfold/share/doc/ && rmdir /opt/doc \
    && ! ldd /opt/rocm/lib/libamdhip64.so.7 | grep 'not found' \
    && ! LD_LIBRARY_PATH=/opt/strix/lib:/opt/rocm/core-10.1/lib ldd /opt/strix/lib/libamdhip64.so.7 | grep 'not found'
# PM4 first; unset LD_LIBRARY_PATH and DEBUG_HIP_GRAPH_PM4 for the stock ROCm 10.0 runtime (/opt/rocm/lib fallback)
ENV TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/native/share/tensorfold/cuda/gfx1151 \
    TF_FLASHNEXT_DEPTH=15 \
    LD_LIBRARY_PATH=/opt/strix/lib:/opt/rocm/core-10.1/lib \
    DEBUG_HIP_GRAPH_PM4=1 \
    GPU_MAX_HW_QUEUES=1 \
    HF_HUB_OFFLINE=1 \
    TENSORFOLD_NO_UPDATE_CHECK=1
WORKDIR /opt/tensorfold
ENTRYPOINT ["tensorfold-native"]
CMD ["capabilities", "--json"]
