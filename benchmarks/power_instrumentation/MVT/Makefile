include ../common.mk
SOURCE := mvt.cu

.PHONY: all
all: mvt

mvt: ${SOURCE}
	${NVCC} ${NVCC_FLAGS} ${INC} ${INST_FLAG} $^ -o $@ ${LIBS}

clean:
	rm -f mvt
