include ../common.mk
SOURCE := 2DConvolution.cu

.PHONY: all
all: 2DConvolution

2DConvolution: ${SOURCE}
	${NVCC} ${NVCC_FLAGS} ${INC} ${INST_FLAG} $^ -o $@ ${LIBS}

clean:
	rm -f 2DConvolution *.out *.bc *.ptx
