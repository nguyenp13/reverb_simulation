#!/usr/bin/python

import scipy.signal
import numpy
import math
from util import *

FILTER_INDEX=0
SIGNAL_COMBINER_INDEX=1

def get_freq_amplitudes(input_signal, sampling_freq):
    # Returns a dict representing the power spectrum. Keys are frequencies. Values are amplitudes. 
    num_samples = len(input_signal)
    freq_res = float(sampling_freq) / num_samples
    return dict(zip([freq_res*i for i in xrange(num_samples)], ((2.0*numpy.absolute(numpy.fft.fft(input_signal))[:int(math.floor(sampling_freq/(freq_res*2.0)))+1])/float(num_samples))))

def get_csd(signal_a, signal_b, sample_rate):
    samples_per_segment = sample_rate#/10 # sample_rate times x makes each segment x seconds
    samples_per_segment = samples_per_segment if samples_per_segment%2==0 else samples_per_segment+1 # to make sure it's odd as the tukey window needs an odd number of samples
    ans = scipy.signal.csd(
        signal_a, 
        signal_b, 
        sample_rate, 
        window=scipy.signal.get_window(('tukey',0.5),samples_per_segment), #Using a tukey window so that it only diminishes volume near the ends of the segment
        nperseg=samples_per_segment,
        noverlap=None, # we set this to None so that it defauls to nperseg/2
        nfft=None, # we set this to None so that it defaults to using nperseg
        detrend='constant', # to remove DC offset
        return_onesided=True, # to remove aliasing above nyquist limit
        scaling='density', # doesn't matter for our purposes since we only care about these values relative to each other
        axis=-1)
    return ans # might have complex values, but they should all be zero since all of our input signals are real-valued

def get_signal_similarity(signal_a, signal_b, sample_rate):
    samples_per_segment = sample_rate#/10 # sample_rate times x makes each segment x seconds
    samples_per_segment = samples_per_segment if samples_per_segment%2==0 else samples_per_segment+1 # to make sure it's odd as the tukey window needs an odd number of samples
    csd = scipy.signal.csd(
        signal_a, 
        signal_b, 
        sample_rate, 
        window=scipy.signal.get_window(('tukey',0.5),samples_per_segment), #Using a tukey window so that it only diminishes volume near the ends of the segment
        nperseg=samples_per_segment,
        noverlap=None, # we set this to None so that it defauls to nperseg/2
        nfft=None, # we set this to None so that it defaults to using nperseg
        detrend='constant', # to remove DC offset
        return_onesided=True, # to remove aliasing above nyquist limit
        scaling='density', # doesn't matter for our purposes since we only care about these values relative to each other
        axis=-1) # might have complex values, but they should all be zero since all of our input signals are real-valued
    score = numpy.sum(numpy.square(csd)[1]) 
    return score

class Filter(object):
    
    def __init__(self, a0=list(), b0=list()): 
        self.a = numpy.array(a0, dtype=numpy.float64) # IIR Filter Coefficients
        self.b = numpy.array(b0, dtype=numpy.float64) # FIR Filter Coefficients
    
    def __init__(self, filter0): 
        self.a = numpy.array(filter0.a)
        self.b = numpy.array(filter0.b)
    
    def apply(self, input_signal):
        output_signal = scipy.signal.lfilter(self.b, self.a, input_signal, axis=0)
        return output_signal
    
class SignalCombiner(object):
    
    def __init__(self, list_of_weights0): 
        self.list_of_weights=numpy.array(list_of_weights0, dtype=numpy.float64)

    def apply(self, list_of_input_signals): 
        output_signal = numpy.sum([weight*input_signal for input_signal, weight in zip(list_of_input_signals,self.list_of_weights)], axis=0)
        return output_signal

class FilterNetwork(object):
    
    def __init__(self, num_layers=3, num_units_per_layer=3, num_fir_coefficients=10000, num_iir_coefficients=10000): 
        # To access a filter unit, we use self.network[layer_index, unit_index]
        self.network = \
                        [
                            [
                                (
                                    Filter([random.uniform(-1.0,1.0) for i in xrange(num_fir_coefficients)],[random.uniform(-1.0,1.0) for i in xrange(num_iir_coefficients)]), 
                                    None if layer_index==0 else SignalCombiner([random.uniform(0.0,1.0) for i in xrange(num_units_per_layer)])
                                )
                                for unit_index in xrange(num_units_per_layer)
                            ] 
                            for layer_index in xrange(num_layers)
                        ]
        self.final_combiner = SignalCombiner([random.uniform(0,1) for i in xrange(num_units_per_layer)])
    
    def get_num_layers(self):
        return len(self.network)
    
    def get_num_units_per_layer(self):
        return len(self.network[0])
    
    def apply(self, input_signal): 
        output_signals_network = [([None]*self.get_num_units_per_layer()) for layer_index in xrange(self.get_num_layers())]
        for layer_index, layer in enumerate(self.network):
            for unit_index, unit in enumerate(layer):
                if layer_index == 0:
                    current_input_signal = input_signal
                else:
                    current_input_signal = unit[SIGNAL_COMBINER_INDEX].apply(output_signals_network[layer_index-1])
                output_signals_network[layer_index][unit_index] = unit[FILTER_INDEX].apply(current_input_signal)
        return self.final_combiner.apply(output_signals_network[-1])
    
#class FilterNetworkGeneticAlgorithm(object):
#    
#    def __init__(self, population_size0=10, num_generations0=10, elite_percent0=0.8): 
#        self.population_size = population_size0
#        self.num_generations = num_generations0
#        self.elite_percent = elite_percent0
#        self.population = [FilterNetwork() for i in xrange(population_size)]
#    
#    def run_generation(self, num_generations_to_run=1):
#        for generation_index in xrange(num_generations_to_run):
#            # FIR Mutations
#            # IIR Mutations
#            # Combiner Mutations
#            # Cross Over

