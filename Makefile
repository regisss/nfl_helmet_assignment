# Install the project dependencies
setup_project:
	git submodule update --init --recursive
	pip install -r yolov5/requirements.txt
	pip install -r requirements.txt
.PHONY: setup_project
