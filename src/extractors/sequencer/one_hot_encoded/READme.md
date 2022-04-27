# Coding of the Sequencer Names

## Vector dimensions

### Base vector
    0: 1 for observed absorbance, 0 else \
    1: 1 if something else than absorbance is observed, else 0 \
    2: 1 for red solution, else 0 \
    3: 1 for green solution, else 0 \
    4: 1 for other solution, else 0 \
    5: 1 if ruler is measuring, else 0 \
    6: 1 if ruler is not measuring, else 0 \
    7: 1 if wavelength is 520, else 0 \
    8: 1 if wavelength is not 520 \
    9: action is on other (laser clicks, ruler, drag restarts, transmittance/absorbance clicks, magnifier movements) \
    10: action is on concentration \
    11: action is on flask \
    12: action is on concentrationlab \
    13: action is on pdf \
    14: break \

### State Action vector
    0: 1 for observed absorbance, 0 else
    1: 1 if something else than absorbance is observed, else 0
    2: 1 for red solution, else 0
    3: 1 for green solution, else 0
    4: 1 for other solution, else 0
    5: 1 if ruler is measuring, else 0
    6: 1 if ruler is not measuring, else 0
    7: 1 if wavelength is 520, else 0
    8: 1 if wavelength is not 520
    9: action is on other (laser clicks, transmittance absorbance clicks, restarts timestamps)
    10: action is on concentration
    11: action is on width
    12: action is wavelength
    13: action is on solution
    14: action is on measuring tools (magnifier and ruler)
    15: action is on concentrationlab
    16: action is on pdf
    17: break


## Encoding

### Encoded LSTM
Each click press and release produces one vector, with 1s in the corresponding entries

### Sampled LSTM
Each *interval*, a vector is taken, corresponding to the state and the action happening at this time. The interval must be small enough such that no action is missed. 

### Seconds LSTM
Each click press and release produces one vector, with the length of the event in the corresponding entries

### Adaptive LSTM
Each click press and releases produces one vector. If the event is larger than an *interval*, the event is duplicated, as many times as it can fill the *interval*.
