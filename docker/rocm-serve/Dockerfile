# TensorFold ROCm/HIP serving image for Qwen3.8 Flash Next on Strix Halo (gfx1151), from public sources only.
# From the repo root (no GPU needed):
#   docker build -t tensorfold-rocm:dev docker/rocm-dev                       # the build stage's toolchain
#   docker build -f docker/rocm-serve/Dockerfile -t tensorfold-rocm:serve .   # --build-arg PM4=0: stock runtime only
# Stages: build (engine + HIP code objects + the Triton AOT set), pm4 (pwilkin's retained-PM4 HIP runtime from
# source; skipped with --build-arg PM4=0), then a Fedora 44 runtime with only the binary, the set, the HIP runtimes'
# library closures and the license notices.
ARG FEDORA=registry.fedoraproject.org/fedora:44@sha256:ba35579e107f26a4c2c000390fb3ff549f3858a9584a6b5a35f7fa51f54de309
ARG DEV_IMAGE=tensorfold-rocm:dev
ARG PM4=1

FROM ${DEV_IMAGE} AS build
ARG JOBS=8
COPY . /tensorfold
WORKDIR /tensorfold
# the Triton AOT set (both specs), where tensorfold-native looks beside its binary
RUN TRITON_CACHE_DIR=/tmp/triton PYTHONPATH=/tensorfold/src nice -n 19 python -B tools/zig/flashnext_aot.py build \
      --target hip --spec zig/tests/cuda/flashnext/kernels.json --spec zig/tests/cuda/flashnext/kernels_int4ar.json \
      --jobs ${JOBS} --out /opt/tensorfold/native/share/tensorfold/cuda/gfx1151 2>&1 | tee /tmp/aot.log \
    && grep -q 'kernels ->' /tmp/aot.log && ! grep -q PROBLEM /tmp/aot.log && rm -rf /tmp/triton
# engine + HIP code objects; hipcc failing exactly these kernels (unused on gfx1151) is expected: docs/rocm/SERVE.md
ARG HIP_UNBUILT="fn_qmm fn_qmm_cluster fn_qmm_prefill fn_roce qmm_group"
RUN nice -n 19 zig build -j${JOBS} -Dgpu=hip -Dhipcc=/opt/rocm/bin/hipcc --prefix /opt/tensorfold native install \
    && got=$(grep -ao 'TF_HIP_BUILD_FAILED [a-z_0-9]*' /opt/tensorfold/native/bin/tensorfold-native \
             | awk 'NF==2{print $2}' | sort -u | xargs) \
    && { [ "$got" = "$HIP_UNBUILT" ] || { echo "hipcc failed: $got (expected: $HIP_UNBUILT)"; exit 1; }; } \
    && rm -rf /root/.cache/zig .zig-cache
RUN d=/opt/tensorfold/share/doc && mkdir -p $d/tensorfold $d/rocm-10.0 \
    && cp -a LICENSE LICENSES NOTICE THIRD_PARTY_NOTICES.md $d/tensorfold/ \
    && cp -a /opt/rocm/core-10.0/share/doc/. $d/rocm-10.0/
# the stock ROCm 10.0 HIP runtime: libamdhip64 + libhiprtc and their /opt closures (comgr, LLVM, rocm_sysdeps)
COPY docker/rocm-serve/closure.sh /usr/local/bin/
RUN l=/opt/rocm/core-10.0/lib && mkdir /out && LD_LIBRARY_PATH= closure.sh /out \
      $l/libamdhip64.so.7 $l/libhiprtc.so.7 $l/libhiprtc-builtins.so.7 \
    && cp -a /opt/rocm/lib /opt/rocm/core /out/opt/rocm/

# pwilkin/rocm-systems' retained-PM4 ROCr + HIP (ilintar-experiments, pinned) on ROCm 10.1, as the halo-box llama
# recipe builds it; PM4=1 only (+5-7% single-stream decode, docs/rocm/SERVE.md)
FROM ${FEDORA} AS pm4
ARG JOBS=8
ARG ROCM_VERSION=10.1
ARG ROCM_RELEASE=10.1.0-3
RUN printf '%s\n' '[amdrocm-stable]' 'name=ROCm' 'baseurl=https://stable.repo.amd.com/rocm/core/packages/rhel10/x86_64' \
      'enabled=1' 'priority=50' 'gpgcheck=1' 'gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg' \
      > /etc/yum.repos.d/rocm.repo \
    && dnf -y --nodocs --setopt=install_weak_deps=False install \
         amdrocm${ROCM_VERSION}-gfx1151-${ROCM_RELEASE} amdrocm-core-devel${ROCM_VERSION}-gfx1151-${ROCM_RELEASE} \
         gcc gcc-c++ make cmake ninja-build git-core python3 python3-pip which vim-common \
         elfutils-libelf-devel libdrm-devel numactl-devel libzstd-devel zlib-ng-compat-devel libglvnd-devel \
    && dnf clean all && rm -rf /var/cache/dnf/* \
    && v=$(basename "$(realpath /opt/rocm/core-${ROCM_VERSION})") && ln -sfn "$v" /opt/rocm/core \
    && for d in bin include lib libexec share; do ln -sfn "$v/$d" "/opt/rocm/$d"; done \
    && ln -sfn "$v/lib/llvm" /opt/rocm/llvm && ln -sfn "$v/lib/llvm/amdgcn" /opt/rocm/amdgcn
ARG ROCM_SYSTEMS_REPO=https://github.com/pwilkin/rocm-systems.git
ARG ROCM_SYSTEMS_BRANCH=ilintar-experiments
ARG ROCM_SYSTEMS_REF=7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1
RUN set -eux; s=/src/rocm-systems; \
    git clone --filter=blob:none --branch "$ROCM_SYSTEMS_BRANCH" "$ROCM_SYSTEMS_REPO" $s; \
    git -C $s -c advice.detachedHead=false checkout "$ROCM_SYSTEMS_REF"; \
    python3 -m venv /src/venv && /src/venv/bin/pip install --no-cache-dir 'CppHeaderParser==2.7.4'; \
    export PATH=/src/venv/bin:$PATH; \
    cmake -S $s/projects/rocr-runtime -B /src/build/rocr -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/src/rocr -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_PREFIX_PATH=/opt/rocm -DBUILD_SHARED_LIBS=ON; \
    nice -n 19 cmake --build /src/build/rocr --parallel $JOBS; cmake --install /src/build/rocr; \
    export LD_LIBRARY_PATH=/src/rocr/lib:/opt/rocm/core/lib:/opt/rocm/core/lib/llvm/lib; \
    cmake -S $s/projects/clr -B /src/build/hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/src/hip -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_PREFIX_PATH="/src/rocr;/opt/rocm" \
      -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd -DHIP_COMMON_DIR=$s/projects/hip \
      -DHIPCC_BIN_DIR=/opt/rocm/bin -DLLVM_ROOT=/opt/rocm/lib/llvm -DClang_ROOT=/opt/rocm/lib/llvm \
      -DROCM_PATH=/src/rocr -Dhsa-runtime64_DIR=/src/rocr/lib/cmake/hsa-runtime64 \
      -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=OFF -DHIP_ENABLE_ROCPROFILER_REGISTER=ON \
      -DUSE_PROF_API=ON -D__HIP_ENABLE_PCH=ON; \
    nice -n 19 cmake --build /src/build/hip --parallel $JOBS; cmake --install /src/build/hip; \
    grep -aq DEBUG_HIP_GRAPH_PM4 /src/hip/lib/libamdhip64.so.7; \
    mkdir -p /opt/strix/lib /opt/strix/doc; \
    cp -a /src/rocr/lib/libhsa-runtime64.so* /src/hip/lib/libamdhip64.so* /src/hip/lib/libhiprtc*.so* /opt/strix/lib/; \
    cp $s/projects/clr/LICENSE.md /opt/strix/doc/LICENSE-clr.md; cp $s/projects/rocr-runtime/LICENSE.txt /opt/strix/doc/LICENSE-rocr.txt; \
    echo "rocm_systems=$(git -C $s rev-parse HEAD) rocm=$ROCM_VERSION-$ROCM_RELEASE" > /opt/strix/doc/versions.txt; \
    rm -rf /src
COPY docker/rocm-serve/closure.sh /usr/local/bin/
RUN l=/opt/strix/lib && mkdir /out && LD_LIBRARY_PATH=$l:/opt/rocm/core-10.1/lib closure.sh /out \
      $l/libamdhip64.so.7 $l/libhiprtc.so.7 $l/libhiprtc-builtins.so.7 \
    && mkdir -p /out/opt/doc && cp -a /opt/strix/doc /out/opt/doc/pm4 && cp -a /opt/rocm/core-10.1/share/doc /out/opt/doc/rocm-10.1

FROM ${FEDORA} AS runtime
# system libraries the runtimes link (the PM4 ROCr takes the distribution's libdrm/libelf/libnuma)
RUN dnf -y --nodocs --setopt=install_weak_deps=False install elfutils-libelf libdrm numactl-libs libzstd libstdc++ \
    && dnf clean all && rm -rf /var/cache/dnf/*
COPY --from=build /out/opt /opt
COPY --from=build /opt/tensorfold /opt/tensorfold
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native \
    && ! ldd /opt/rocm/lib/libamdhip64.so.7 | grep 'not found'
ENV TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/native/share/tensorfold/cuda/gfx1151 \
    TF_FLASHNEXT_DEPTH=15 \
    HF_HUB_OFFLINE=1 \
    TENSORFOLD_NO_UPDATE_CHECK=1
WORKDIR /opt/tensorfold
ENTRYPOINT ["tensorfold-native"]
CMD ["capabilities", "--json"]

# PM4=0: the stock ROCm 10.0 runtime (the engine falls back to /opt/rocm/lib/libamdhip64.so.7)
FROM runtime AS runtime-pm4-0
ENV DEBUG_HIP_GRAPH_PM4=0

# PM4=1: the PM4 libs first on the loader path. Stock runtime instead, at run time: LD_LIBRARY_PATH="" and
# DEBUG_HIP_GRAPH_PM4=0 (and GPU_MAX_HW_QUEUES=4, HIP's default)
FROM runtime AS runtime-pm4-1
COPY --from=pm4 /out/opt /opt
RUN mv /opt/doc/* /opt/tensorfold/share/doc/ && rmdir /opt/doc \
    && ! LD_LIBRARY_PATH=/opt/strix/lib:/opt/rocm/core-10.1/lib ldd /opt/strix/lib/libamdhip64.so.7 | grep 'not found'
ENV LD_LIBRARY_PATH=/opt/strix/lib:/opt/rocm/core-10.1/lib \
    DEBUG_HIP_GRAPH_PM4=1 \
    GPU_MAX_HW_QUEUES=1

FROM runtime-pm4-${PM4}
