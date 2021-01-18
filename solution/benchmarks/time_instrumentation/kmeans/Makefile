include ../common.mk
SOURCE := main.cu kernels.cu

.PHONY: all
all: kmeans

kmeans: ${SOURCE}
	${NVCC} ${NVCC_FLAGS} ${INC} ${INST_FLAG}  $^ -o $@ ${LIBS}

clean:
	rm -f kmeans

