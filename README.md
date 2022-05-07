# Pipeline Lab
This repository helps implementing the whole pipeline used to process simulation logs and classify students into different groups according to their conceptual understanding.

It uses k-fold nested cross validation, and can do online and offline predictions on full and partial sequences.

## Current state
This is the implementation of the general pipeline-lab. There is a branch for the beerslaw-lab and one for the capacitor lab.
https://phet.colorado.edu/sims/html/beers-law-lab/latest/beers-law-lab_en.html

## Structure
```bash
├── data
|	├── experiment keys
|	├── raw
|	├── parsed simulations
|	├── sequenced simulations
└── experiments
└── reports
└── src
|   ├── batches
|	├── configs
|	├── extractors
|	|	├── aggregator
|	|	├── cleaners
|	|	├── concatenator
|	|	├── encoding
|	|	├── lengths
|	|	├── parser
|	|	├── sequencer
|	├── logs
|	├── ml
|	|	├── gridsearches
|	|	├── models
|	|	├── samplers
|	|	├── scorers
|	|	├── splitters
|	|	├── xvalidators
|	├── notebooks
|	├── utils
|	├── visualisers
└── wikis
```	




