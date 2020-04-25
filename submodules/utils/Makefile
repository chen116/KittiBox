SHELL := /bin/bash

.PHONY: all-python2 all-python3
all-python2:
	pip install --no-cache-dir runcython
	makecython++ stitch_wrapper.pyx "" "stitch_rects.cpp ./hungarian/hungarian.cpp"

all-python3:
	pip3 install --no-cache-dir runcython3
	makecython3++ stitch_wrapper.pyx "" "stitch_rects.cpp ./hungarian/hungarian.cpp"

