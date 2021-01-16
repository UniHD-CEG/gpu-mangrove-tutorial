#!/usr/bin/env bash

# Exit on error
set -e

echo building cuda flux binaries..
(cd benchmarks/cuda_flux && make)
echo building time instrumentation binaries..
(cd benchmarks/time_instrumentation && make)
echo building power instrumentation binaries..
(cd benchmarks/power_instrumentation && make)

echo copy binaries..
mkdir -p measurements/my_benchmarks/bin/flux
cp benchmarks/cuda_flux/kmeans/kmeans measurements/my_benchmarks/bin/flux/.
cp benchmarks/cuda_flux/5p-stencil/5p-stencil measurements/my_benchmarks/bin/flux/.
cp benchmarks/cuda_flux/5p-stencil/5p-stencil_opt measurements/my_benchmarks/bin/flux/.
cp benchmarks/cuda_flux/2DCONV/2DConvolution measurements/my_benchmarks/bin/flux/.
cp benchmarks/cuda_flux/ATAX/atax measurements/my_benchmarks/bin/flux/.
cp benchmarks/cuda_flux/COVAR/covariance measurements/my_benchmarks/bin/flux/.
cp benchmarks/cuda_flux/GEMM/gemm measurements/my_benchmarks/bin/flux/.
cp benchmarks/cuda_flux/MVT/mvt measurements/my_benchmarks/bin/flux/.

mkdir -p measurements/my_benchmarks/bin/time
cp benchmarks/time_instrumentation/kmeans/kmeans measurements/my_benchmarks/bin/time/.
cp benchmarks/time_instrumentation/5p-stencil/5p-stencil measurements/my_benchmarks/bin/time/.
cp benchmarks/time_instrumentation/5p-stencil/5p-stencil_opt measurements/my_benchmarks/bin/time/.
cp benchmarks/time_instrumentation/2DCONV/2DConvolution measurements/my_benchmarks/bin/time/.
cp benchmarks/time_instrumentation/ATAX/atax measurements/my_benchmarks/bin/time/.
cp benchmarks/time_instrumentation/COVAR/covariance measurements/my_benchmarks/bin/time/.
cp benchmarks/time_instrumentation/GEMM/gemm measurements/my_benchmarks/bin/time/.
cp benchmarks/time_instrumentation/MVT/mvt measurements/my_benchmarks/bin/time/.

mkdir -p measurements/my_benchmarks/bin/power
cp benchmarks/power_instrumentation/kmeans/kmeans measurements/my_benchmarks/bin/power/.
cp benchmarks/power_instrumentation/5p-stencil/5p-stencil measurements/my_benchmarks/bin/power/.
cp benchmarks/power_instrumentation/5p-stencil/5p-stencil_opt measurements/my_benchmarks/bin/power/.
cp benchmarks/power_instrumentation/2DCONV/2DConvolution measurements/my_benchmarks/bin/power/.
cp benchmarks/power_instrumentation/ATAX/atax measurements/my_benchmarks/bin/power/.
cp benchmarks/power_instrumentation/COVAR/covariance measurements/my_benchmarks/bin/power/.
cp benchmarks/power_instrumentation/GEMM/gemm measurements/my_benchmarks/bin/power/.
cp benchmarks/power_instrumentation/MVT/mvt measurements/my_benchmarks/bin/power/.
