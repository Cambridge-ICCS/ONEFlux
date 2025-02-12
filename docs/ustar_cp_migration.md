# Migrating ustar_cp from MATLAB to Python

In 2024-25, a team from the Institute of Computing for Climate Science undertook to translate the MATLAB implementation of the ustar_cp step of ONEFlux into Python 3. This document summarises the approach and provides a final 'retirement plan' for the MATLAB code.

Team at Cambridge:

* Isaac Akanho
* James Emberton
* Dominic Orchard
* Tianzhang Cai

The work also leveraged an initial translation by Peter Isaac (OzFlux).

With thanks to discussion and input also from Gilberto Pastorello (Lawrence Berkeley Labs) and Omar Jamil (ICCS, Cambridge).

## Migration methodology

We follow a test-driven approach to migration to ensure, as far as possible, semantic preservation from the MATLAB to Python. Our approach had three steps:

1. Modularise MATLAB code into smaller function components;

2. Write language-agnostic tests in Python for all functions, which can then be applied to the MATLAB code. Test approaches included:
    a. Smoke tests
    b. Unit tests
    c. Property-based tests
    d. Data-driven tests generated from site data.

3. Traverse the dependency graph of the MATLAB code from leaf to root, translating each function in turn and ensuring that the tests all pass on the translated code.

In some cases additional 'differential' tests were employed, generating random data and comparing the MATLAB and Python implementations.

The actual translation process combined a number of techniques:
  a. Using the initial hand translation by Peter Isaac;
  b. Using an in-house extended version of the [https://github.com/victorlei/smop/](libsmop) tool, called [MatoPy](https://github.com/tztsai/MatoPy);
  c. Using LLMs;
  d. Hand translation.

## Resulting code structure

Within the top-level directory we have:

- `oneflux_steps/ustar_cp` - Original MATLAB
- `oneflux_steps/ustar_cp_refactor` - Modularised MATLAB code
- `oneflux_steps/ustar_cp_python` - Python translation

We have also added

- `tests/conftest.py` - Test engine
- `tests/unit_tests/test_ustar_cp` - Extensive test suite for `ustar_cp`
- `tests/test_artefacts` - Test fixtures mostly comprising site data and intermediate input-output data generated from site data

Tests can be run for the MATLAB code (in `ustar_cp_refactor`) by running at the top-level:

        pytest tests/unit_tests/test_ustar_cp --language=matlab

and for the Python translation by running:

        pytest tests/unit_tests/test_ustar_cp --language=python

## Multi-language test suite

We provide a language-agnostic test suite that can switch between MATLAB (using the [matlab.engine FFI](https://uk.mathworks.com/help/matlab/matlab-engine-for-python.html) for connecting Python to MATLAB) and Python code. This approach allows the same set of tests to be run against both MATLAB and Python implementations, ensuring consistency and correctness across different languages.

The core of this functionality is provided by test fixtures in `tests/conftest.py`. Here, an abstract base class `TestEngine` defines the language-agnostic interface
against which instances of the class provide MATLAB and Python test runners. 

The TestEngine abstract base class defines the following methods that need to be implemented by any concrete test engine:

* `_repr_pretty_`: A placeholder method that enables the Hypothesis package (a property-based testing framework) to work with this runner as a fixture;

* `convert`: Converts the input to a type and format compatible with the engine. 

* `unconvert`: Can be used if there is a need to invert `convert` (although this rarely needed);

* `equal`: Compares two values for equality in the representation used by the engine.

Two concrete implementations are provided inheriting from the abstract base class: `PythonEngine` and `MatlabEngine`. Crucially the `MatlabEngine` wraps the Python-MATLAB interface and handles calling functions in the MATLAB code, mapping any MATLAB errors to Python exceptions, and converting the result to a Python form (i.e., we typically do not need to `convert` the result).

The following is an example unit test written for `pytest` using the `test_engine` fixture provided by `conftest.py`. The code tests the `funName` function:

```
def test_function(test_engine):
    input_data = test_engine.convert([1, 2, 3])
    result = test_engine.funName(input_data)
    expected = [2, 3, 4]
    assert test_engine.equal(result, expected)
```

This test can then be run against any test engine to target the requisite language. The language can then be switched by passing the command-line argument `--language=LANG` to `pytest` where `LANG` is either `python` or `matlab` (the default at the moment).

# MATLAB Retirement Plan

For now, we preserve the MATLAB code alongside the Python. The following explains how to finally remove the MATLAB from the code base, including converting the test suite to be Python only.

## Remove differential tests

The differential tests involve invoking Python and MATLAB and comparing the results. 
These are in `tests/unit_tests/ustar_cp` and have file names of the form `test_differential_` and so can just be removed via

        git rm ests/unit_tests/ustar_cp/test_differential_*.py

## Remove MATLAB engine test engine and its dependencies

* From `requirements.txt` remove the `matlabengine` line.
* From `conftest.py` remove any code that is between comments `# <MATLAB>` and `# </MATLAB>` delineating code for matlab test engines.
* OPTIONAL: remove 'matlab' from the `get_languages():` function

## Switch oneflux_steps to point to ustar_cp_python

Target `launch` function in `oneflux_steps/ustar_cp/python/launch.py`

## Remove the MATLAB code

Delete the `oneflux_steps/ustar_cp` and `oneflux_steps/ustar_cp_recator_wip` folders
