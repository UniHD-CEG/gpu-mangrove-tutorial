include ../common.mk
SOURCE := covariance.cu

.PHONY: all
all: covariance

covariance: ${SOURCE}
	${NVCC} ${NVCC_FLAGS} ${INC} ${INST_FLAG} $^ -o $@ ${LIBS}

clean:
	rm -f covariance *.out *.bc *.ptx
