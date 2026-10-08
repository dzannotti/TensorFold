# TensorFold ROCm/HIP dev image for Strix Halo (gfx1151).
# Base: Fedora 44 + ROCm 10.0 (HIP 7.15) for gfx1151, with gcc/cmake/ninja/git/python3-devel,
# hipcc/amdclang++, hipblaslt, hipify-perl/clang, rocprofv3, rocm-smi already in /opt/rocm.
FROM local/ciru-toolchain:rocm10.0

RUN dnf -y --nodocs --setopt=install_weak_deps=False install perl xz tar \
    && dnf clean all && rm -rf /var/cache/dnf/*

# Zig 0.17.0 (exact; build.zig refuses others). sha256 from ziglang.org/download/index.json.
ARG ZIG_VERSION=0.17.0
ARG ZIG_SHA256=1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026
RUN curl -fsSL -o /tmp/zig.tar.xz https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz \
    && echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - \
    && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
    && ln -s /opt/zig/zig /usr/local/bin/zig && rm /tmp/zig.tar.xz

# PyTorch + Triton for gfx1151 from AMD's TheRock stable index (bundles its own ROCm 7.13 runtime
# via the rocm-sdk-* wheels; /opt/rocm stays HIP 7.15 for hipcc/native builds).
ARG ROCM_WHL_INDEX=https://repo.amd.com/rocm/whl/gfx1151/
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --index-url ${ROCM_WHL_INDEX} \
         "torch==2.11.0+rocm7.13.0" "triton==3.6.0+rocm7.13.0" \
    && /opt/venv/bin/pip install --no-cache-dir numpy safetensors huggingface_hub transformers ninja
# venv after /opt/rocm/bin: the rocm-sdk wheels drop their own hipcc/amdclang++/rocprofv3 shims (ROCm 7.13)
# into /opt/venv/bin; /opt/rocm (HIP 7.15) must win. venv python3 still precedes /usr/bin.
ENV PATH=/opt/rocm/bin:/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:/opt/venv/bin:/usr/local/bin:/usr/bin \
    PYTORCH_ROCM_ARCH=gfx1151

WORKDIR /home/dzannotti/tf-rocm
CMD ["/bin/bash"]
