include ../common.mk
SOURCE := main.cu kernels.cu

.PHONY: all
all: 5p-stencil 5p-stencil_opt

5p-stencil: ${SOURCE}
	${NVCC} ${NVCC_FLAGS} ${INC} $^ -o $@ ${LIBS}

5p-stencil_opt: ${SOURCE}
	${NVCC} ${NVCC_FLAGS} ${INC} -DOPT_KERNEL $^ -o $@ ${LIBS} 

clean:
	rm -f 5p-stencil 5p-stencil_opt *.bc *.out *.ptx
