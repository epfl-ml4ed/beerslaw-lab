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
from visualisers.timelines.KateTimeline import KateTimeline
from visualisers.timelines.seri_timeline import SeriTimeline

def generate_all_timelines(settings):
    timeliner = KateTimeline(settings)
    simulations = os.listdir(settings['paths']['crawl_path'])
    simulations = [sim for sim in simulations if 'simulation.pkl' in sim]
    for sim in simulations:
        print(sim)
        with open(settings['paths']['crawl_path'] + sim, 'rb') as fp:
            sim = pickle.load(fp)
        timeliner.create_timeline(sim)

def sort_all_timelines(settings):
    # labels
    label_map = '../data/beerslaw/experiment_keys/permutation_maps/vector_binary.yaml'
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

def test(settings):
    simulations = os.listdir('../data/beerslaw/parsed simulations/')
    usernames = [
        'perm2013_lidsvdphyjs_t2v_simulation'
    ]
    files = []
    for username in usernames:
        files = files + [sim for sim in simulations if username in sim and ('t1' in sim or 't2' in sim or 't3' in sim)]
    simulations = ['../data/beerslaw/parsed simulations/' + file for file in files]
    
    for sim_path in simulations:
        with open(sim_path, 'rb') as fp:
            sim = pickle.load(fp)

        timeliner = KateTimeline(settings)
        timeliner.create_timeline(sim)


def main(settings):
    if settings['test']:
        test(settings)

    if settings['crawl']:
        generate_all_timelines(settings)

    if settings['sort']:
        sort_all_timelines(settings)



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
    parser.add_argument('--savepng', dest='savepng', default=False, help='save timelines in svg formats', action='store_true')
    parser.add_argument('--show', dest='show', default=False, help='show timelines', action='store_true')

    settings.update(vars(parser.parse_args()))
    
    main(settings)
