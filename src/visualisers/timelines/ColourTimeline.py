import os
from bokeh.models.tools import Toolbar
import yaml
import json
import pickle
from typing import Tuple

import numpy as np

import bokeh
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, HoverTool, glyph
from bokeh.plotting import figure, output_file, show, save
from bokeh.io import export_svg, export_png


from extractors.parser.simulation_parser import Simulation
from extractors.sequencer.flat.plotter_sequencer import PlotterSequencing

from visualisers.timelines.Timeline import Timeline

class ColourTimeline(Timeline):
    """
    This class is created to plot the parsed files as timelines, to represent student interactions in a human-readable way.
    Particularly, this timeline is aimed at reading parsed files from the beer's law lab [https://phet.colorado.edu/sims/html/beers-law-lab/latest/beers-law-lab_en.html]
    
    The two top bars will display the colour of the solution and the wavelength respectively
    The middle bar will show the interactions
    The two bottom bars will display whether there was measurement from the ruler, and then from the absorbance/transmittance/nomeasured
    """
    def __init__(self, settings: dict):
        self._name = 'timeline'
        self._notation = 'tmln'
        self._settings = settings
        
        self._load_palette()
        self._load_plotter()
        

    def _load_palette(self):
        with open('./visualisers/maps/colourtimeline_cm.yaml', 'rb') as fp:
            self._palette = yaml.load(fp, Loader=yaml.FullLoader)

    def _load_plotter(self):
        self._plotter = PlotterSequencing()

    def _wavelength_to_rgb(self, wavelength, gamma=0.8):
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

    def _plot_wavelength(self, sim: Simulation, glyphs:dict, plot):
        """
        Plot the colour of the wavelength as the top bar state
        Args:
            sim: simulation to plot
            plot: plot object
        """
        values, ts = sim.get_wavelength()
        values.append(values[-1])
        ts.append(ts[-1])
        rect_width, middle_coord = self._extract_values_xs(ts)
        wavelength_colours = [self._wavelength_to_rgb(v) for v in values]
        wavelength_colours = wavelength_colours[:-1]
        y_coord = self._settings['timeline']['base_coord']
        y_coord += self._settings['timeline']['action_height'] / 2 # action bar
        y_coord += self._settings['timeline']['state_height'] # concentration bar
        y_coord += self._settings['timeline']['state_height'] / 2 # Finding the middle y coordinate
        y_coords = [y_coord for l in rect_width]
        heights = [self._settings['timeline']['state_height'] for _ in y_coords]

        source = {
            'x': middle_coord,
            'y': y_coords,
            'w': rect_width,
            'h': heights,
            'colour': wavelength_colours,
            'label': values[:-1]

        }
        glyphs['wavelength'] = plot.rect(x='x', y='y', width='w', height='h', fill_color='colour', line_alpha=0, source=source, legend_label='wavelength')
        plot.add_tools(HoverTool(renderers=[glyphs['wavelength']], tooltips=[('wavelength', "@label")]))
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
        solution_colours = [s.replace('beersLawLab.beersLawScreen.solutions.', '') for s in values]
        solution_colours = [self._palette['solution_colours'][sol] for sol in solution_colours]
        solution_colours = solution_colours[:-1]
        y_coord = self._settings['timeline']['base_coord']
        y_coord += self._settings['timeline']['action_height'] / 2 # action bar
        y_coord += self._settings['timeline']['state_height'] / 2 # Finding the middle y coordinate
        y_coords = [y_coord for l in rect_width]
        heights = [self._settings['timeline']['state_height'] for _ in y_coords]

        source = {
            'x': middle_coord,
            'y': y_coords,
            'w': rect_width,
            'h': heights,
            'colour': solution_colours,
            'label': values[:-1]
        }
        glyphs['solution'] = plot.rect(x='x', y='y', width='w', height='h', fill_color='colour', line_alpha=0, source=source, legend_label='solution')
        plot.add_tools(HoverTool(renderers=[glyphs['solution']], tooltips=[('solution', "@label")]))
        return glyphs, plot

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
        begins, ends, labels = sequencer.get_sequences(sim, sim.get_learner_id())
        # for i in range(len(begins)):
        #     print(begins[i], ends[i], labels[i])
        labels_to_plot = [
            'laser', 'ruler', 'restarts', 'transmittance_absorbance', 'magnifier', 
            'wavelength', 'solution', 'concentration', 'flask', 'pdf', 'concentrationlab', 'wavelength_slider'

        ]

        bs = {}
        es = {}
        for label in labels_to_plot:
            bs[label], es[label] = self._retrieve_beginends(label, begins, ends, labels)
            widths, middles = self._extract_beginsends_xs(bs[label], es[label])
            colours = [self._palette['timeline'][label] for _ in range(len(middles))]
            y_coords = [self._settings['timeline']['base_coord'] for _ in range(len(colours))]
            heights = [self._settings['timeline']['action_height'] for _ in y_coords]

            source = {
                'x': middles,
                'y': y_coords,
                'w': widths,
                'h': heights,
                'colour':colours,
                'label': [label for _ in colours]
            }

            glyphs[label] = plot.rect(x='x', y='y', width='w', height='h', fill_color='colour', line_alpha=0, source=source, legend_label=label)
            plot.add_tools(HoverTool(renderers=[glyphs[label]], tooltips=[('action', "@label")]))
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
        colours = [self._palette['timeline']['ruler'] for _ in range(len(middle_coords))]
        y_coord = self._settings['timeline']['base_coord']
        y_coord -= self._settings['timeline']['action_height'] / 2
        y_coord -= self._settings['timeline']['state_height'] / 2
        y_coords = [y_coord for _ in range(len(colours))]
        heights = [self._settings['timeline']['state_height'] for _ in y_coords]

        source = {
            'x': middle_coords,
            'y': y_coords,
            'w': widths,
            'h': heights,
            'colour': colours,
            'label': ['rule measuring' for _ in colours]
        }
        glyphs['ruler measuring'] = plot.rect(x='x', y='y', width='w', height='h', fill_color='colour', source=source, legend_label='ruler measuring', line_alpha=0)
        plot.add_tools(HoverTool(renderers=[glyphs['ruler measuring']], tooltips=[('ruler measuring', "@label")]))

        # invisible_source = ColumnDataSource(data=dict(
        #         x=[sim.get_last_timestamp()/2], y=[y_coord],
        #         w=[sim.get_last_timestamp()], h=[self._settings['timeline']['state_height']],
        #         color=['white']
        #         ))
        # glyphs['blank ruler'] = plot.rect(x='x', y='y', width='w', height='h', fill_color='color', fill_alpha=0, line_alpha=0, source=invisible_source)
        return glyphs, plot

    def _plot_measuring(self, sim:Simulation, glyphs, plot, plotter: PlotterSequencing):
        """Plot whether the absorbance, the transmittance, or nothing was measured

        Args:
            sim (Simulation): simulation for whyoat to do this for
            glyphs ([type]): dict[label, object] that contains all the glyphs and their labels
            plot: plot object
            plotter ([type]): sequencer for the plotter in order to reuse code from previous steps
        """

        values, timesteps = plotter.get_absorbance_transmittance_nothing(sim)
        values = [values[0]] + values
        timesteps = [0] + timesteps
        #debug
        for i, val in enumerate(values):
            print(timesteps[i], val)
        
        values.append(values[-1])
        timesteps.append(sim.get_last_timestamp())
        widths, coords = self._extract_values_xs(timesteps)
        colours = [self._palette['timeline'][label] for label in values[:-1]]
        y_coord = self._settings['timeline']['base_coord']
        y_coord -= self._settings['timeline']['action_height'] / 2
        y_coord -= self._settings['timeline']['state_height']
        y_coord -= self._settings['timeline']['state_height'] / 2
        y_coords = [y_coord for _ in range(len(colours))]
        heights = [self._settings['timeline']['state_height'] for _ in y_coords]

        source = {
            'x': coords,
            'y': y_coords,
            'w': widths,
            'h': heights,
            'colour': colours,
            'label': values[:-1]
        }
        glyphs['dependent variable'] = plot.rect(x='x', y='y', width='w', height='h', fill_color='colour', source=source, legend_label='dependent variable', line_alpha=0)
        plot.add_tools(HoverTool(renderers=[glyphs['dependent variable']], tooltips=[('dependent variable', "@label")]))

        # invisible_source = ColumnDataSource(data=dict(
        #     x=[sim.get_last_timestamp()/2], y=[y_coord],
        #     w=[sim.get_last_timestamp()], h=[self._settings['timeline']['state_height']],
        #     color=['white']
        #     ))
        # glyphs['blank dependent variable'] = plot.rect(x='x', y='y', width='w', height='h', fill_color='color', fill_alpha=0, line_alpha=0, source=invisible_source)
        return glyphs, plot

    def _frame_timeline(self, glyphs, plot, sim:Simulation):
        y_coord = self._settings['timeline']['base_coord']
        height = self._settings['timeline']['state_height']
        height += self._settings['timeline']['state_height']
        height +=self._settings['timeline']['action_height']
        height += self._settings['timeline']['state_height']
        height += self._settings['timeline']['state_height']
        width = sim.get_last_timestamp()

        source = ColumnDataSource(dict(
            x=[sim.get_last_timestamp()/2], y=[y_coord],
            w=[width], h=[height],
            color=['white']
            ))
        plot.rect(x='x', y='y', width='w', height='h', fill_color='color', fill_alpha=0, line_color='black', line_alpha=1, source=source)
        return glyphs, plot

    def create_timeline(self, sim: Simulation):
        glyphs = {}
        title = 'Timeline for student ' + sim.get_learner_id() + ' with permutation ' + sim.get_permutation() + ' for task ' + str(sim.get_task())
        plot = self._init_figure(title)

        glyphs, plot = self._plot_wavelength(sim, glyphs, plot)
        glyphs, plot = self._plot_solutioncolour(sim, glyphs, plot)
        glyphs, plot = self._plot_actions(sim, glyphs, plot, self._plotter)
        glyphs, plot = self._plot_ruler(sim, glyphs, plot, self._plotter)
        glyphs, plot = self._plot_measuring(sim, glyphs, plot, self._plotter)
        glyphs, plot = self._frame_timeline(glyphs, plot, sim)

        plot.legend.click_policy="hide"

        if self._settings['savepng']:
            # plot.output_backend = 'png'
            path = '../reports/' + self._settings['image']['report_folder']
            path += '/colour timelines/'
            os.makedirs(path, exist_ok=True)
            path += 'p' + sim.get_permutation() 
            path += '_l' + sim.get_learner_id()
            path += '_t' + str(sim.get_task()) + '.png'
            print(path)
            export_png(plot, filename=path)

        if self._settings['saveimg']:
            plot.output_backend = 'svg'
            path = '../reports/' + self._settings['image']['report_folder']
            path += '/colour timelines/'
            os.makedirs(path, exist_ok=True)
            path += 'p' + sim.get_permutation() 
            path += '_l' + sim.get_learner_id()
            path += '_t' + str(sim.get_task()) + '.svg'
            print(path)
            export_svg(plot, filename=path)

        if self._settings['save']:
            path = '../reports/' + self._settings['image']['report_folder']
            path += '/colour timelines/'
            os.makedirs(path, exist_ok=True)
            path += 'p' + sim.get_permutation() 
            path += '_l' + sim.get_learner_id()
            path += '_t' + str(sim.get_task()) + '.html'
            save(plot, filename=path)

        if self._settings['show']:
            show(plot)







        
        











