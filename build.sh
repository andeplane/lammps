#!/bin/bash

# Build script for LAMMPS with Kokkos enabled (threads backend) using Emscripten

set -e  # Exit on error

source "/Users/anderhaf/repos/emsdk/emsdk_env.sh"

# Clean previous build
rm -rf build_kokkos

# Create build directory and configure
mkdir build_kokkos && cd build_kokkos

# Configure with emcmake (Emscripten CMake wrapper)
# Enable 64-bit pointers for Kokkos compatibility (MEMORY64=2 uses 64-bit pointers but wasm32 binary for compatibility)
# Add MEMORY64=2 to compiler flags so CMake detects 8-byte pointers during configuration
emcmake cmake ../cmake \
  -DPKG_KOKKOS=ON \
  -DKokkos_ENABLE_THREADS=ON \
  -DKokkos_ENABLE_LIBDL=OFF \
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=OFF \
  -DKokkos_ENABLE_DEBUG=OFF \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_FLAGS="-sMEMORY64=2" \
  -DCMAKE_C_FLAGS="-sMEMORY64=2"

# Now add pthread to compiler flags for the actual build
EMSCRIPTEN_COMPILE_FLAGS="-pthread -sMEMORY64=2"

emcmake cmake \
  -DCMAKE_CXX_FLAGS="${EMSCRIPTEN_COMPILE_FLAGS}" \
  -DCMAKE_C_FLAGS="${EMSCRIPTEN_COMPILE_FLAGS}" \
  .

echo ""
echo "Configuration complete!"
echo ""
echo "Emscripten linker flags have been automatically added to the lmp target."
echo ""
echo "To build: cd build_kokkos && make"

