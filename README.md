# Gauss map

`gaussmap` is a Python package that generates
[Manim](https://www.manim.community) animations showing the
[Gauss map](https://en.wikipedia.org/wiki/Gauss_map) transformation of a
parameteric surface in 3D.

## Dependencies

The dependencies of `gaussmap` are listed in the `pyproject.toml` file's
`[project]` section under `dependencies`. Additionally, the supported Python
versions are listed in the `pyproject.toml` file.

Optional dependencies for testing, code style, the Jupyter notebook can be
found in the `pyproject.toml` file's `[project.optional-dependencies]` section.

Make sure to check your distribution's package manager before using `pip` to
install the dependencies. You can find the name of your distribution's package
using [Repology](https://repology.org).

Some of the important dependencies to look up are

- [Manim](https://repology.org/project/manim)
- [NumPy](https://repology.org/project/python:numpy)
- [SymPy](https://repology.org/project/python:sympy)
- [Notebook](https://repology.org/project/python:notebook)
- [pytest](https://repology.org/project/python:pytest)
- [pytest-cov](https://repology.org/project/python:pytest-cov)
- [Mypy](https://repology.org/project/mypy)

## Installing

To install the package locally clone this repo and run pip install.

```sh
git clone https://github.com/jtckr/gaussmap
cd gaussmap
python -m pip install .
```

## Running

To run one of the example scenes use the `run-scene` shell script.
```sh
./run-scene ./scenes/CylinderScene.py
```

To use your own custom parameteric surface run
```sh
./run-scene ./scenes/CustomScene.py
```

A list of options for the `run-scene` script can be found by running
```sh
./run-scene -h
```

## Testing

Install the test dependencies by running
```sh
python -m pip install .[test,coverage,style]
```

Then run the tests using
```sh
pytest ./tests
```

To do code coverage reports run
```sh
pytest --cov=gaussmap ./tests
```

Type hints can be checked using
```sh
mypy gaussmap tests scenes
```

## Notebook

Install the Jupyter notebook dependencies by running
```sh
python -m pip install .[notebook]
```

Then run the Jupyter notebook using
```sh
jupyter notebook gaussmap.ipynb
```
the Jupyter notebook should then open in your default web browser

## Contributing

If you would like to contribute, please fork the repository and make a GitHub
pull request. Pull requests are warmly welcome!

## Links

- [Repository](https://github.com/jtckr/gaussmap)
- [Issue tracker](https://github.com/jtckr/gaussmap/issues)

## Acknowledgements

Thanks to Grant Sanderson and the Manim community developers for making
[Manim](https://www.manim.community).

Manfredo P. do Carmo's *Differential Geometry of Curves and Surfaces* 2nd
edition was used as a reference for the Gauss map calculations.

Thanks to
[Carl Friedrich Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) for
discovering the Gauss map and its applications.
