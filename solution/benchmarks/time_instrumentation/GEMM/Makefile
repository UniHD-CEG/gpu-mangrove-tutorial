include ../common.mk
SOURCE := gemm.cu

.PHONY: all
all: gemm

gemm: ${SOURCE}
	${NVCC} ${NVCC_FLAGS} ${INC} ${INST_FLAG} $^ -o $@ ${LIBS}

clean:
	rm -f gemm
