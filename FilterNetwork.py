#!/usr/bin/python

import scipy.signal
from util import *

FILTER_INDEX=0
SIGNAL_COMBINER_INDEX=1

class Filter(object):
    
    def __init__(self, a0=list(), b0=list()): 
        self.a = numpy.array(a0, dtype=numpy.float64) # IIR Filter Coefficients
        self.b = numpy.array(b0, dtype=numpy.float64) # FIR Filter Coefficients
    
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
    
    def mutate_combiner(self, layer_index0 = None, unit_index0 = None):
        if self.get_num_layers() == 1: 
            return
        layer_index = random.randint(1,self.get_num_layers()-1) if layer_index0 == None else layer_index0
        unit_index = random.randint(0,self.get_num_units_per_layer()-1) if unit_index0 == None else unit_index0
        combiner = self.network[layer_index][unit_index][SIGNAL_COMBINER_INDEX]
        num_weights = len(combiner.list_of_weights)
        combiner.list_of_weights[random.randint(0,num_weights-1)] = random.uniform(0.0,1.0)
    
    def mutate_FIR(self, layer_index0 = None, unit_index0 = None):
        layer_index = random.randint(0,self.get_num_layers()-1) if layer_index0 == None else layer_index0
        unit_index = random.randint(0,self.get_num_units_per_layer()-1) if unit_index0 == None else unit_index0
        filt = self.network[layer_index][unit_index][FILTER_INDEX]
        num_coefficients = len(filt.b)
        filt.b[random.randint(0,num_coefficients-1)] = random.uniform(0.0,1.0)
    
    def mutate_IIR(self, layer_index0 = None, unit_index0 = None):
        layer_index = random.randint(0,self.get_num_layers()-1) if layer_index0 == None else layer_index0
        unit_index = random.randint(0,self.get_num_units_per_layer()-1) if unit_index0 == None else unit_index0
        filt = self.network[layer_index][unit_index][FILTER_INDEX]
        num_coefficients = len(filt.a)
        filt.a[random.randint(0,num_coefficients-1)] = random.uniform(0.0,1.0)
    
