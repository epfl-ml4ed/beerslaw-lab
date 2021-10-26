import os
import re
import shutil
from shutil import copyfile

import yaml
import pickle
import argparse
import logging 

import numpy as np
import pandas as pd

from visualisers.timelines.Timeline import Timeline
from visualisers.timelines.ColourTimeline import ColourTimeline

def generate_all_timelines(settings):
    timeliner = ColourTimeline(settings)
    simulations = os.listdir(settings['paths']['crawl_path'])
    for sim in simulations:
        print(sim)
        with open(settings['paths']['crawl_path'] + sim, 'rb') as fp:
            sim = pickle.load(fp)
        timeliner.create_timeline(sim)

def sort_all_timelines(settings):
    # labels
    label_map = '../data/experiment keys/permutation_maps/vector_binary.yaml'
    with open(label_map) as fp:
            label_map = yaml.load(fp, Loader=yaml.FullLoader)
    # paths
    old_path = settings['paths']['timelines_path']
    new_path = lambda label: settings['paths']['new_path'] + label + '/' 

    # timelines
    permutation_finder = re.compile('p([0-9]+)')
    timelines = os.listdir(settings['paths']['timelines_path'])
    for timeline in timelines:
        print(timeline)
        try:
            ranking = permutation_finder.findall(timeline)[0]
            label = label_map['map'][ranking]
        except IndexError:
            label = 'missing ranking'
        except KeyError:
            label = 'error'
        source = old_path + timeline
        direction = new_path(label) + timeline
        copyfile(source, direction)

def update_rankings(settings):
    parsed_simulations = os.listdir(settings['paths']['crawl_path'])
    with open('../data/post_test/rankings.pkl', 'rb') as fp:
        rankings = pickle.load(fp)
        rankings = rankings.set_index('username')

    for sim in parsed_simulations:
        with open(settings['paths']['crawl_path'] + sim, 'rb') as fp:
            simu = pickle.load(fp)
        try:
            permu = rankings.loc[simu.get_learner_id()]
            simu.set_permutation(permu['ranking'])
        except KeyError:
            simu.set_permutation('missing')
        
def test(settings):
    with open('../data/parsed simulations/perm_lid4k4dk9pu_t1v_simulation.pkl', 'rb') as fp:
        sim = pickle.load(fp)

    timeliner = ColourTimeline(settings)
    timeliner.create_timeline(sim)

def main(settings):
    if settings['test']:
        test(settings)

    if settings['crawl']:
        generate_all_timelines(settings)

    if settings['sort']:
        sort_all_timelines(settings)

    if settings['updateranking']:
        update_rankings(settings)


if __name__ == '__main__':
    with open('./configs/timeline_conf.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Plot Timelines')
    # actions
    parser.add_argument('--test', dest='test', default=False, help='make tests on the timeline', action='store_true')
    parser.add_argument('--crawl', dest='crawl', default=False, help='create all visual timelines', action='store_true')
    parser.add_argument('--sort', dest='sort', default=False, help='put them in folders according to ranking', action='store_true')
    parser.add_argument('--updateranking', dest='updateranking', default=False, help='update the rankings on the simulation with the new data', action='store_true')
    

    # plot flags
    parser.add_argument('--save', dest='save', default=False, help='save timelines in html formats', action='store_true')
    parser.add_argument('--saveimg', dest='saveimg', default=False, help='save timelines in svg formats', action='store_true')
    parser.add_argument('--show', dest='show', default=False, help='show timelines', action='store_true')

    settings.update(vars(parser.parse_args()))
    
    main(settings)
