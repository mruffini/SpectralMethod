# SpectralMethod

Python implementation of the algorithms described in the paper "A new spectral method for Latent Variable Models", by Matteo Ruffini, Marta Casanellas and Ricard Gavaldà.

The paper can be found here:

    https://arxiv.org

## Contents of the project:

### RandomGenerator.py

A module that allows the user to generate synthetic texts distributed as a the Single Topic Model or as a Latent Dirichlet Allocation.

### SpectralMehod.py

A module that contains the learning algorithms described in the paper; the user given a set of texts, expressed in the form of a term-document matrix, can use these algorithms to learn the latent variable models generating the texts (in the case of the Single topic model and Latent Dirichlet Allocation).

### TestScript.py

A script to generate synthetic texts using the module "RandomGenerator" and learn the hidden model parameters using the module "SpectralMethod".
