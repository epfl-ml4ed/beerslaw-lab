import yaml
import json
import pickle
from typing import Tuple

import numpy as np

import bokeh
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Rect
from extractors.parser.simulation_parser import Simulation
from extractors.sequencer.plotter_sequencer import PlotterSequencing

class Timeline:
    """
    This class is created to plot the parsed files as timelines, to represent student interactions in a human-readable way.
    Particularly, this timeline is aimed at reading parsed files from the beer's law lab [https://phet.colorado.edu/sims/html/beers-law-lab/latest/beers-law-lab_en.html]
    
    The two top bars will display the colour of the solution and the wavelength respectively
    The middle bar will show the interactions
    The two bottom bars will display whether there was measurement from the ruler, and then from the absorbance/transmittance/nomeasured
    """
    def __init__(self, settings: dict, plotter: PlotterSequencing):
        self._name = 'timeline'
        self._notation = 'tmln'
        self._settings = settings

    def _load_palette(self):
        with open('./visualisers/maps/colourtimeline_cm.yaml', 'rb') as fp:
            self._palette = yaml.load(fp, Loader=yaml.FullLoader)

    def _extract_xs(self, component):
        """
        From a component of the simulation, extract the timecoordinates for 
        the rectangle in bokeh.
        
        Args:
            component: list (of timestamps) or dict (on and off)
        """
        raise NotImplementedError
    
    def _extract_values_xs(self, timestamps: list) -> Tuple[list, list]:
        """
        Returns 
            - the middle coordinates for the rectangle function in bokeh
            - the width of each of the rectangles to be drawn
        """
        ts0 = [x for x in timestamps[:-1]]
        ts1 = [x for x in timestamps[1:]]

        rect_width = np.array(ts1) - np.array(ts0)
        middle_coord = (rect_width / 2) + np.array(ts0)
        return rect_width, middle_coord

    def _wavelength_to_rgb(wavelength, gamma=0.8):
        # From http://www.noah.org/wiki/Wavelength_to_RGB_in_Python

        '''
            == A few notes about color ==
            Color   Wavelength(nm) Frequency(THz)
            Red     620-750        484-400
            Orange  590-620        508-484
            Yellow  570-590        526-508
            Green   495-570        606-526
            Blue    450-495        668-606
            Violet  380-450        789-668
            f is frequency (cycles per second)
            l (lambda) is wavelength (meters per cycle)
            e is energy (Joules)
            h (Plank's constant) = 6.6260695729 x 10^-34 Joule*seconds
                                = 6.6260695729 x 10^-34 m^2*kg/seconds
            c = 299792458 meters per second
            f = c/l
            l = c/f
            e = h*f
            e = c*h/l
            List of peak frequency responses for each type of 
            photoreceptor cell in the human eye:
                S cone: 437 nm
                M cone: 533 nm
                L cone: 564 nm
                rod:    550 nm in bright daylight, 498 nm when dark adapted. 
                        Rods adapt to low light conditions by becoming more sensitive.
                        Peak frequency response shifts to 498 nm.
        '''

        '''This converts a given wavelength of light to an 
        approximate RGB color value. The wavelength must be given
        in nanometers in the range from 380 nm through 750 nm
        (789 THz through 400 THz).
        Based on code by Dan Bruton
        http://www.physics.sfasu.edu/astro/color/spectra.html
        '''

        wavelength = float(wavelength)
        if wavelength >= 380 and wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif wavelength >= 440 and wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif wavelength >= 490 and wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** gamma
        elif wavelength >= 510 and wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif wavelength >= 580 and wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** gamma
            B = 0.0
        elif wavelength >= 645 and wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
        else:
            R = 0.0
            G = 0.0
            B = 0.0
        R *= 255
        G *= 255
        B *= 255
        return (int(R), int(G), int(B))

    def _plot_wavelength(self, sim: Simulation, glyphs, plot):
        """
        Plot the colour of the wavelength as the top bar state
        Args:
            sim: simulation to plot
            plot: plot object
        """
        values, ts = sim.get_wavelength()
        rect_width, middle_coord = self._extract_values_xs(ts)
        wavelength_colours = [self._wavelength_to_rgb(v) for v in values]
        wavelength_colours = wavelength_colours[:-1]
        y_coord = self._settings['plot']['base_coord']
        y_coord += self._settings['plot']['action_height'] / 2 # action bar
        y_coord += self._settings['plot']['state_height'] # concentration bar
        y_coord += self._settings['plot']['state_height'] / 2 # Finding the middle y coordinate
        y_coords = [y_coord for l in rect_width]

        source = ColumnDataSource(dict(
            x=middle_coord, y=y_coords, 
            w=rect_width, h=self._settings['plot']['state_height']),
            colour=wavelength_colours
        )
        glyphs['object'].append(Rect(x='x', y='y', width='w', height='h', fill_color='colour'))
        glyphs['label'].append('wavelength')
        plot.add_glyph(source, glyphs['object'][-1])
        return glyphs, plot

    def _plot_solutioncolour(self, sim: Simulation, glyphs, plot):
        """
        Plot the colour of the solution below the wavelength bar
        Args: 
            sim: simualtion to parse
            glyphs: dict[label, object] that contains all the glyphs and their labels
            plot: plot object
        """
        values, ts = sim.get_solution()
        rect_width, middle_coord = self._extract_values_xs(ts)
        solution_colours = [self._palette['solution_colours'][sol] for sol in values]
        solution_colours = solution_colours[:-1]
        y_coord = self._settings['plot']['base_coord']
        y_coord += self._settings['plot']['action_height'] / 2 # action bar
        y_coord += self._settings['plot']['state_height'] / 2 # Finding the middle y coordinate
        y_coords = [y_coord for l in rect_width]

        source = ColumnDataSource(dict(
            x=middle_coord, y=y_coords, 
            w=rect_width, h=self._settings['plot']['state_height']),
            colour=solution_colours
        )
        glyphs['object'].append(Rect(x='x', y='y', width='w', height='h', fill_color='colour'))
        glyphs['label'].append('wavelength')
        plot.add_glyph(source, glyphs['object'][-1])
        return glyphs, plot

    def _retrieve_beginends(self, label:str, begins:list, ends:list, labels:list) -> Tuple[list, list]:
        """Gives the beginning and end timestamps of a specific label
        Args:
            label (str): label of interest
            begins (list): beginning timestamps of all labels
            ends (list): end timestamps of all labels
            labels (list): list of labels

        Returns:
            Tuple[list, list]: 
                bs: beginning timestamps of all the action of label
                es: end timestamps of all the action of label
        """
        indices = [i for i in range(len(labels)) if label == labels[i]]
        bs = [begins[idx] for idx in indices]
        es = [ends[idx] for idx in indices]
        return bs, es

    def _extract_beginsends_xs(self, begins:dict, ends:dict) -> Tuple[list, list]:
        """Returns the middle points of the bars, as well as their width (time)

        Args:
            begins (dict): begin timestamps
            ends (dict): end timestamps

        Returns:
            Tuple[list, list]: 
                widths:
        """
        widths = np.array(ends) - np.array(begins)
        middle_coords = (np.array(widths) / 2) + np.array(begins)
        return widths, middle_coords
 
    def _plot_actions(self, sim:Simulation, glyphs, plot, plotter: PlotterSequencing):
        """
        Plot all of the actions from the beer's law lab:
        - ticks from the checkboxes, 
        - aquarium width
        - laser toggle + change of light
        - solution selection
        - concentration
        - ruler moves
        - transmittance clicks
        """
        
        sequencer = PlotterSequencing()
        begins, ends, labels = sequencer.get_sequences(sim)

        labels = [
            'laser', 'ruler', 'restarts', 'transmittance_absorbance', 'magnifier_position', 
            'wavelength', 'solution', 'concentration', 'flask', 'pdf', 'concentrationlab'

        ]

        bs = {}
        es = {}
        for label in labels:
            bs[label], es[label] = self._retrieve_beginends(label, begins, ends, labels)
            widths, middles = self._extract_beginsends_xs(bs[label], es[label])
            colours = [self._palette['timeline']['label'] for _ in range(len(middles))]
            y_coords = [self._settings['plot']['base_coord'] for _ in range(len(colours))]

            source = ColumnDataSource(dict(
                x=middles, y=y_coords, 
                w=widths, h=self._settings['plot']['action_height']),
                colour=colours
            )
            glyphs['object'].append(Rect(x='x', y='y', width='w', height='h', fill_color='colour'))
            glyphs['label'].append(label)
            plot.add_glyph(source, glyphs['object'][-1])
        return glyphs, plot

    def _plot_ruler(self, sim:Simulation, glyphs, plot, plotter: PlotterSequencing):
        """Plot whether the ruler was measuring something or not, below the actino bar

        Args:
            sim (Simulation): simulation to get the ruler measure from
            glyphs dict[label, object] that contains all the glyphs and their labels
            plot: plot object
            plotter ([type]): sequencer for the plotter in order to reuse code from previous steps
        """
        begin_ends_ruler = plotter.get_ruler_timepoints(sim)
        widths, middle_coords = self._extract_beginsends_xs(begin_ends_ruler['begin'], begin_ends_ruler['end'])
        colours = [self._palette['timeline']['label'] for _ in range(len(middle_coords))]
        y_coord = self._settings['plot']['base_coord']
        y_coord -= self._settings['plot']['action_height'] / 2
        y_coord -= self._settings['plot']['state_height'] / 2
        y_coords = [y_coord for _ in range(len(colours))]

        source = ColumnDataSource(dict(
            x=middle_coords, y=y_coords, 
            w=widths, h=self._settings['plot']['action_height']),
            colour=colours
        )
        glyphs['object'].append(Rect(x='x', y='y', width='w', height='h', fill_color='colour'))
        glyphs['label'].append('ruler')
        plot.add_glyph(source, glyphs['object'][-1])
        return glyphs, plot

    def _plot_measuring(self, sim:Simulation, glyphs, plot, plotter: PlotterSequencing):
        """Plot whether the absorbance, the transmittance, or nothing was measured

        Args:
            sim (Simulation): simulation for what to do this for
            glyphs ([type]): dict[label, object] that contains all the glyphs and their labels
            plot: plot object
            plotter ([type]): sequencer for the plotter in order to reuse code from previous steps
        """




        
        













    def _create_timeline(self):
        """
        Creates the timeline
        """
        raise NotImplementedError
        

    