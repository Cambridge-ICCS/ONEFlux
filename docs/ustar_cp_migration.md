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

The migration methodology the team employed was:

1. Modularise MATLAB code into smaller function components;
2. Write tests language agnostic tests in Python for all functions, using the Python matlab FFI (matlab-engine) to apply the tests to MATLAB. Test approaches included:
   a. Smoke tests
   b. Unit tests
   c. Property-based tests
   d. Data-driven tests generated from site data.
3. Traverse the depenency graph of the MATLAB code from leaf to root,
translating each function in turn and ensuring that the Python tests
pass.
   In some cases additional 'differential' tests were employed,
   generating random data and comparing the MATLAB and python implementations.


   