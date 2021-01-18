include ../common.mk
SOURCE := atax.cu

.PHONY: all
all: atax

atax: ${SOURCE}
	${NVCC} ${NVCC_FLAGS} ${INC} ${INST_FLAG} $^ -o $@ ${LIBS}

clean:
	rm -f atax
