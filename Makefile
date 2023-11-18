pypi: clean build upload

build:
	python3 -m build
upload:
	python3 -m twine upload dist/*
clean:
	rm -rf dist/
sdist:
	python -m build . --sdist
