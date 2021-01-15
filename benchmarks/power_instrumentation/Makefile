.PHONY: all
all: kmeans 5p-stencil 2DConv atax covar gemm mvt

.PHONY: kmeans
kmeans:
	cd kmeans && make

.PHONY: 5p-stencil
5p-stencil:
	cd 5p-stencil && make

.PHONY: 2DConv
2DConv:
	cd 2DCONV && make

.PHONY: atax
atax:
	cd ATAX && make

.PHONY: covar
covar:
	cd COVAR && make

.PHONY: gemm
gemm:
	cd GEMM && make

.PHONY: mvt
mvt:
	cd MVT && make

clean: kmeans_clean 5p-stencil_clean 2DConv_clean atax_clean covar_clean gemm_clean mvt_clean

.PHONY: kmeans_clean
kmeans_clean:
	cd kmeans && make clean

.PHONY: 5p-stencil_clean
5p-stencil_clean:
	cd 5p-stencil && make clean

.PHONY: 2DConv_clean
2DConv_clean:
	cd 2DCONV && make clean

.PHONY: atax_clean
atax_clean:
	cd ATAX && make clean

.PHONY: covar_clean
covar_clean:
	cd COVAR && make clean

.PHONY: gemm_clean
gemm_clean:
	cd GEMM && make clean

.PHONY: mvt_clean
mvt_clean:
	cd MVT && make clean

