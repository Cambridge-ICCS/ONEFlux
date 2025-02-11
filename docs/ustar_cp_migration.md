# Migrating ustar_cp from MATLAB to Python

In 2024, a team from the Institute of Computing for
Climate Science undertook to translate the MATLAB implementation 
of ustar_cp into Python 3. This document summarises the
approach and provides a final 'retirement plan' for the
MATLAB code.

Team:
* Isaac Akanho
* James Emberton
* Dominic Orchard
* Tianzhang Cai

The work also leveraged an initial translation by Peter Isaac (OzFlux).

## Migration methodology

1. Modularise MATLAB code into smaller function components;
2. Write tests language agnostic tests in Python for all functions, which can then be applied
to the MATLAB code. Test approaches included:
   a. Smoke tests
   b. Unit tests
   c. Property-based tests
   d. Data-driven tests generated from site data.
3. Traverse the dependency graph of the MATLAB code from leaf to root,
translating each function in turn and ensuring that the Python tests
pass.

In some cases additional 'differential' tests were employed,
generating random data and comparing the MATLAB and python implementations.

The actual translation process combined a number of techniques:
  a. Using the inital hand-translation by Peter Isaac;
  b. Using an in-house extended version of the [https://github.com/victorlei/smop/](libsmop) tool, called [Matopy](https://github.com/tztsai/MatoPy).
  c. Using LLMs
  d. Hand translation

## Multi-language test suite

We provide a language-agnostic test suite that can switch between MATLAB (using the matlab.engine FFI
for connecting Pythont to MATLAB) and Python code. This approach allows the same set of tests to be run against both MATLAB and Python implementations, ensuring consistency and correctness across different languages.

In `contest.py` an abstract base class `TestEngine` defines the language-agnostic interface
against which instances of the class provide MATLAB and Python test runners. 

The TestEngine abstract base class defines the following methods that need to be implemented by any concrete test engine:

* `_repr_pretty_`: A placeholder method that enables the Hypothesis package (a property-based testing framework) to work with this runner as a fixture;
* `convert`: Converts the input to a type and format compatible with the engine. 
* `unconvert`: Can be used if there is a need to invert `convert` (although this rarely needed);
* `equal`: Compares two values for equality in the representation used by the engine.

Two concrete implementations are provided inherting
from the abstract base class: `PythonEngine`
and `MatlabEngine`. 

Here is an example usage in a simple `pytest` unit test:

```
def test_function(test_engine):
    input_data = [1, 2, 3]
    result = test_engine.convert(input_data)
    expected = test_engine.convert([2, 3, 4])
    assert test_engine.equal(result, expected)
```

The language can then be switched by passing the
command-line argument `--language=LANG` to `pytest`
where `LANG` is either `python` or `matlab` (the default
at the moment).

# Retirement Plan

For now, we preserve the MATLAB code alongside the Python.
The following explains how to finally remove the MATLAB
and convert the test suite to be Python only.
