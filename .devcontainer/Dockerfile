FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
    chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    lld ccache \
    python3 python3-pip python3-dev \
    clangd clang-format
RUN python3 -m pip install --break-system-packages pre-commit

WORKDIR /vulkan
ARG VULKAN_VER=1.4.328.1
RUN wget https://sdk.lunarg.com/sdk/download/${VULKAN_VER}/linux/vulkansdk-linux-x86_64-${VULKAN_VER}.tar.xz
RUN tar xf vulkansdk-linux-x86_64-${VULKAN_VER}.tar.xz
RUN apt-get install -y  --no-install-recommends libxcb-xinput0 libxcb-xinerama0 libxcb-cursor-dev
RUN echo "source /vulkan/${VULKAN_VER}/setup-env.sh" >> /etc/bash.bashrc

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
RUN chmod +555 /root
RUN chmod +555 /root/.local/bin
ENV PATH="/root/.local/bin/:$PATH"
