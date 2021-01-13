.PHONY: all
all: kmeans 5p-stencil

.PHONY: kmeans
kmeans:
	cd kmeans && make

.PHONY: 5p-stencil
5p-stencil:
	cd 5p-stencil && make

clean: kmeans_clean 5p-stencil_clean

.PHONY: kmeans_clean
kmeans_clean:
	cd kmeans && make clean

5p-stencil_clean:
	cd 5p-stencil && make clean

