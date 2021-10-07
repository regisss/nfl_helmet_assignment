# Install the project dependencies
setup_project:
	git submodule update --init --recursive
	cd yolov5
	pip install -r requirements.txt
	cd ..
	pip install -r requirements.txt
.PHONY: setup_project
