# TensorFold ROCm/HIP dev image for Strix Halo (gfx1151), from public sources only:
# Fedora 44 + AMD's ROCm 10.0 (HIP 7.15) gfx1151 packages (hipcc, amdclang++, hipblaslt, hipify, rocprofv3),
# gcc/cmake/ninja/python3-devel, Zig 0.17.0, and AMD's torch/Triton wheels for gfx1151.
#   docker build -t tensorfold-rocm:dev docker/rocm-dev
ARG FEDORA=registry.fedoraproject.org/fedora:44@sha256:ba35579e107f26a4c2c000390fb3ff549f3858a9584a6b5a35f7fa51f54de309
FROM ${FEDORA}

# AMD's signed stable repo; the release is pinned (10.0.0-4 is what the port was qualified with)
ARG ROCM_VERSION=10.0
ARG ROCM_RELEASE=10.0.0-4
RUN printf '%s\n' '[amdrocm-stable]' 'name=ROCm' 'baseurl=https://stable.repo.amd.com/rocm/core/packages/rhel10/x86_64' \
      'enabled=1' 'priority=50' 'gpgcheck=1' 'gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg' \
      > /etc/yum.repos.d/rocm.repo \
    && dnf -y --nodocs --setopt=install_weak_deps=False install \
         amdrocm${ROCM_VERSION}-gfx1151-${ROCM_RELEASE} amdrocm-core-devel${ROCM_VERSION}-gfx1151-${ROCM_RELEASE} \
         gcc gcc-c++ make cmake ninja-build git-core curl perl xz tar which patch python3 python3-devel python3-pip \
    && dnf clean all && rm -rf /var/cache/dnf/* \
    && v=$(basename "$(realpath /opt/rocm/core-${ROCM_VERSION})") && ln -sfn "$v" /opt/rocm/core \
    && for d in bin include lib libexec share; do ln -sfn "$v/$d" "/opt/rocm/$d"; done \
    && ln -sfn "$v/lib/llvm" /opt/rocm/llvm && ln -sfn "$v/lib/llvm/amdgcn" /opt/rocm/amdgcn \
    && hipcc --version | grep -q 'HIP version: 7.15'

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
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:/opt/venv/bin:/usr/local/bin:/usr/bin \
    LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib \
    PYTORCH_ROCM_ARCH=gfx1151

CMD ["/bin/bash"]
