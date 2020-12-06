#!/usr/bin/env bash

# Exit on error
set -e

echo building cuda flux binaries..
(cd benchmarks/cuda_flux && make)
echo building time instrumentation binaries..
(cd benchmarks/time_instrumentation && make)

echo copy binaries..
mkdir -p measurements/mybenchmarks/bin/flux
cp benchmarks/cuda_flux/kmeans/kmeans measurements/mybenchmarks/bin/flux/.
cp benchmarks/cuda_flux/5p-stencil/5p-stencil measurements/mybenchmarks/bin/flux/.
cp benchmarks/cuda_flux/5p-stencil/5p-stencil_opt measurements/mybenchmarks/bin/flux/.

mkdir -p measurements/mybenchmarks/bin/time
cp benchmarks/time_instrumentation/kmeans/kmeans measurements/mybenchmarks/bin/time/.
cp benchmarks/time_instrumentation/5p-stencil/5p-stencil measurements/mybenchmarks/bin/time/.
cp benchmarks/time_instrumentation/5p-stencil/5p-stencil_opt measurements/mybenchmarks/bin/time/.
