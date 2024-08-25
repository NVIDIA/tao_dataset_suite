all: build install

build:
	python3 setup.py bdist_wheel

clean:
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	rm -rf ./**/*/__pycache__
	rm -rf .**/*/__pycache__

install: build
	pip3 install dist/nvidia_tao_ds-*.whl

develop:
	python3 setup.py develop

uninstall:
	pip3 uninstall -y nvidia-tao-ds

