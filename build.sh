#!/bin/bash

# Build script for LAMMPS with Kokkos enabled (threads backend) using Emscripten

set -e  # Exit on error

# Clean previous build
rm -rf build_kokkos

# Create build directory and configure
mkdir build_kokkos && cd build_kokkos

# Configure with emcmake (Emscripten CMake wrapper)
# Enable 64-bit pointers for Kokkos compatibility (MEMORY64=2 uses 64-bit pointers but wasm32 binary for compatibility)
# Configure without linker flags to avoid interfering with CMake test compilations
emcmake cmake ../cmake \
  -DPKG_KOKKOS=ON \
  -DKokkos_ENABLE_THREADS=ON \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_FLAGS="-sMEMORY64=2" \
  -DCMAKE_C_FLAGS="-sMEMORY64=2"

# Now add Emscripten linker flags
# The first configuration cached all test results (including Kokkos C++17 check)
# This second configuration should use cached results and not re-run tests
EMSCRIPTEN_FLAGS="-sMEMORY64=2 -s NO_DISABLE_EXCEPTION_CATCHING=1 -s ALLOW_MEMORY_GROWTH=1 -s ASYNCIFY -s EXPORTED_RUNTIME_METHODS=[\"getValue\",\"FS\",\"HEAP32\",\"HEAPF32\",\"HEAPF64\",\"allocateUTF8\",\"stringToUTF8\",\"UTF8ToString\",\"_malloc\",\"_free\",\"setValue\"] -s EXPORT_NAME=createModule -s FORCE_FILESYSTEM=1 -s EXPORTED_FUNCTIONS=[\"_lammps_open_no_mpi\",\"_lammps_close\",\"_lammps_commands_string\",\"_lammps_file\",\"_lammps_command\"]"

emcmake cmake -DCMAKE_EXE_LINKER_FLAGS="${EMSCRIPTEN_FLAGS}" .

echo "Configuration complete. Run 'cd build_kokkos && make' to build."

