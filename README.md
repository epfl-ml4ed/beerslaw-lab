# phet-capacitor-lab
# How to adapt this code for new simulations

1. Create, for each student, a dictionary comprising (can use sequencer for easiness if the parser is similar to the capacitor lab):
- the sequence of actions [key: sequence] -> list
- the starting timestamps [key: begin] -> list
- the end timestamps [key: end] -> list
- the permutation [key: permutation] -> str
- the last timestamp of the permutation (end point, last (non)event recorded) [key: last_timestamp] -> float
- the unique id [key: learner_id] -> str

*ex: {sequence:['action1', 'action2', 'action1'], start:[0, 1, 2], end:[0.5, 1.5, 2.5], permutation: 1234, learner_id: 239598}*

2. Save each of those files individually
- preferrably choose the following format for the path: ../data/sequenced simulations/ + name of the action group + /p_ + permutation + _lid + learner_id
 + _sequenced.pkl

3. Create a dictionary
- longueur minimum des séquences [key: limit] -> int
- information about the sequences [key: sequences] -> dict
	For each sequence, create a dictionary:
	- the index [it's the key]
	- the path where the simulation was saved [key: path] -> str
	- the length of the sequence [key: length] -> int
	- the unique id of the learner [key: learner_id] -> str
- index of the map unique id -> index [key: index] -> dict
	- key: learner id
	- value: index
- save it in the format of ../data/sequenced simulations/ + name of the action group + /id_dictionary.pkl

4. Create a sequencer
	- use the n17 states template
	- you only need to create the list of state
	- if the parser was similar to the phet capacitor, use the functions in a script to create your data once and for all

4. Open pipeline_maker.py
	a. edit _choose_sequencer(self):
		- add the sequencer in the imports and as an if statement

5. In utils.config_handler
- add the class type + class number

