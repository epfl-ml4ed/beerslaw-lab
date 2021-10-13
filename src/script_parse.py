import os
import pickle
import yaml
import pickle
import argparse
import logging

import numpy as np
import pandas as pd

from extractors.parser.example_parser import Simulation as CapacitorSimulation
from extractors.parser.simulation_parser import Simulation as ChemlabSimulation
from extractors.sequencer.sequencing import Sequencing

def parse_simulations(settings: dict):
    print('hello')
    """Parses all the logs and transforms them into objects into the folder ../../data/parsed simulations/

    Args:
        settings (dict): settings read from the yaml
        
    Flag: --parse
    """
    log_path = './logs/parsing_chemlab.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO, 
        format='', 
        datefmt=''
    )
    
    if settings['capacitor']:
        Simulation = CapacitorSimulation
        
    elif settings['chemlab']:
        Simulation = ChemlabSimulation
        
    repo = '//ic1files.epfl.ch/D-VET/Projects/ChemLab/04_Processing/Data Backups/'
    files = []
    for r, d, f in os.walk(repo):
        for file in f:
            if file.endswith(".log"):
                files.append(os.path.join(r, file))
                
    # print(files)
    with open('../data/debug/parsed.pkl', 'rb') as fp:
        parsed = pickle.load(fp)
        files = [file for file in files if file not in parsed]
        
        
    for path in files:
        sim = ChemlabSimulation(path)
        if sim.nevents != 0:
            sim.parse_simulation()
            sim.save()
            
            timeline = sim.get_timeline()
            active = sim.get_active_timeline()
            logging.info('TIMELINE')
            for i in range(len(timeline[0])):
                logging.info('action: {}, timestamp: {}'.format(timeline[0][i], timeline[1][i]))
                
            logging.info('')
            logging.info('ACTIVE TIMELINE')
            for i in range(len(active[0])):
                logging.info('action: {}, timestamp: {}'.format(active[0][i], active[1][i]))
                
            parsed.append(path)
            
            with open('../data/debug/parsed.pkl', 'wb') as fp:
                pickle.dump(parsed, fp)
        
def test(settings):
    log_path = './logs/parsing_debug.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO, 
        format='', 
        datefmt=''
    )
    
    paths = ['//ic1files.epfl.ch/D-VET/Projects/ChemLab/04_Processing/Data Backups/Session 1/2pqdkrkw-1.log']
    sim = ChemlabSimulation(paths[0])
    sim.parse_simulation()
    
    timeline = sim.get_timeline()
    active = sim.get_active_timeline()
    logging.info('TIMELINE')
    for i in range(len(timeline[0])):
        logging.info('action: {}, timestamp: {}'.format(timeline[0][i], timeline[1][i]))
        
    logging.info('')
    logging.info('ACTIVE TIMELINE')
    for i in range(len(active[0])):
        logging.info('action: {}, timestamp: {}'.format(active[0][i], active[1][i]))
        
    
    logging.info('wavelength {}'.format(sim.get_wavelength()))
    logging.info('width {}'.format(sim.get_width()))
    logging.info('concentration {}'.format(sim.get_concentration()))
    logging.info('solution {}'.format(sim.get_solution()))
    logging.info('light {}'.format(sim.get_light()))
    logging.info('measure {}'.format(sim.get_measure_display()))
    logging.info('recorded {}'.format(sim.get_measure_recorded()))
    
def test_measure_recorded(settings):
    repo = '//ic1files.epfl.ch/D-VET/Projects/ChemLab/04_Processing/Data Backups/'
    files = []
    for r, d, f in os.walk(repo):
        for file in f:
            if file.endswith(".log"):
                files.append(os.path.join(r, file))
    files = [file for file in files if settings['test_id'] in file]
    print('files', files)
    
    sim = ChemlabSimulation(files[0])
    sim.parse_simulation()
    
    # print(sim.get_measure_recorded())
    print(sim.get_ruler())
    print(sim._ruler_position)
    
    # sim = ChemlabSimulation(files[1])
    # sim.parse_simulation()
    
    # sim = ChemlabSimulation(files[2])
    # sim.parse_simulation()
    
    

def main(settings):
    if settings['parse']:
        parse_simulations(settings)
    if settings['test']:
        test(settings)
    if settings['test_measure']:
        test_measure_recorded(settings)

if __name__ == '__main__':
    with open('./configs/parsing_conf.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Logs / Simulations manipulations')
    parser.add_argument('--parse', dest='parse', default=False, action='store_true')
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    parser.add_argument('--testmeas', dest='test_measure', default=False, action='store_true')
    
    parser.add_argument('--chemlab', dest='chemlab', default=False, action='store_true')
    parser.add_argument('--capacitor', dest='capacitor', default=False, action='store_true')
    parser.add_argument('--testid', dest='test_id', default='', action='store')
    settings.update(vars(parser.parse_args()))
        
    main(settings)
    