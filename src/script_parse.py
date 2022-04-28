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
    """Parses all the logs and transforms them into objects into the folder ../../data/beerslaw/parsed simulations/

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
        
    # files to parse
    repo = '../data/beerslaw/temp/'
    files = []
    for r, d, f in os.walk(repo):
        for file in f:
            if file.endswith(".log"):
                files.append(os.path.join(r, file))
                
    # Making sure we do not parse something we have already parsed (in case of bug, makes sure we do not need)
    # to go through the whole database again
    with open('../data/beerslaw/debug/parsed.pkl', 'rb') as fp:
        parsed = pickle.load(fp)
        files = [file for file in files if file not in parsed]

    # Rankings
    with open('../data/beerslaw/post_test/rankings.pkl', 'rb') as fp:
        rankings = pickle.load(fp)
        rankings = rankings.set_index('username')
        
    for path in files:
        print(path)
        sim = ChemlabSimulation(path)
        if sim.nevents != 0:
            sim.parse_simulation()

            try:
                permutation = rankings.loc[sim.get_learner_id()]['ranking']
            except KeyError:
                permutation = 'missing'
            sim.set_permutation(permutation)

            sim.save()
            parsed.append(path)
            
            with open('../data/beerslaw/debug/parsed.pkl', 'wb') as fp:
                pickle.dump(parsed, fp)
        
def update_rankings(settings):
    parsed_simulations = os.listdir(settings['paths']['crawl_path'])
    with open('../data/beerslaw/post_test/rankings.pkl', 'rb') as fp:
        rankings = pickle.load(fp)
        rankings = rankings.set_index('username')

    for sim in parsed_simulations:
        if sim == '.DS_Store':
            continue
        with open(settings['paths']['crawl_path'] + sim, 'rb') as fp:
            simu = pickle.load(fp)
        try:
            permu = rankings.loc[simu.get_learner_id()]
            simu.set_permutation(permu['ranking'])
        except KeyError:
            simu.set_permutation('missing')
        simu.save()
        

def test(settings):
    log_path = './logs/parsing_debug.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO, 
        format='', 
        datefmt=''
    )
    paths = [
        # '../data/beerslaw/temp/Session 8/fj5tdybn-1.log',
        '../data/beerslaw/temp/Session 28/mzjq6z9t-2.log',
        # '../data/beerslaw/temp/Session 8/fj5tdybn-3.log'
    ]
    with open('../data/beerslaw/post_test/rankings.pkl', 'rb') as fp:
        rankings = pickle.load(fp)
        rankings = rankings.set_index('username')

    for path in paths:
        print(path)
        sim = ChemlabSimulation(path)
        print(rankings.loc[sim.get_learner_id()]['ranking'])
        sim.parse_simulation()
        sim.set_permutation(rankings.loc[sim.get_learner_id()]['ranking'])
        print('save')
        sim.save(path='../data/beerslaw/temp parsed/perm'+ sim.get_permutation() + '_lid' + path.split('/')[-1])
        print()



    # sim = ChemlabSimulation(paths[0])
    # sim.parse_simulation()
    
    # timeline = sim.get_timeline()
    # active = sim.get_active_timeline()
    # logging.info('TIMELINE')
    # for i in range(len(timeline[0])):
    #     logging.info('action: {}, timestamp: {}'.format(timeline[0][i], timeline[1][i]))
        
    # logging.info('')
    # logging.info('ACTIVE TIMELINE')
    # for i in range(len(active[0])):
    #     logging.info('action: {}, timestamp: {}'.format(active[0][i], active[1][i]))
        
    
    # logging.info('wavelength {}'.format(sim.get_wavelength()))
    # logging.info('width {}'.format(sim.get_width()))
    # logging.info('concentration {}'.format(sim.get_concentration()))
    # logging.info('solution {}'.format(sim.get_solution()))
    # logging.info('light {}'.format(sim.get_light()))
    # logging.info('measure {}'.format(sim.get_measure_display()))
    # logging.info('recorded {}'.format(sim.get_measure_recorded()))
    
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

    if settings['update_rankings']:
        update_rankings(settings)

if __name__ == '__main__':
    with open('./configs/parsing_conf.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Logs / Simulations manipulations')
    parser.add_argument('--parse', dest='parse', default=False, action='store_true')
    parser.add_argument('--updaterankings', dest='update_rankings', help='input rankings onto permutations', default=False, action='store_true')
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    parser.add_argument('--testmeas', dest='test_measure', default=False, action='store_true')
    
    parser.add_argument('--chemlab', dest='chemlab', default=False, action='store_true')
    parser.add_argument('--capacitor', dest='capacitor', default=False, action='store_true')
    parser.add_argument('--testid', dest='test_id', default='', action='store')
    settings.update(vars(parser.parse_args()))
        
    main(settings)
    