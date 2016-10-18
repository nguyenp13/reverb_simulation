#!/usr/bin/python

import scipy.signal
from util import *

FILTER_INDEX=0
SIGNAL_COMBINER_INDEX=1

class Filter(object):
    
    def __init__(self, a0=list(), b0=list()): 
        self.a = a0
        self.b = b0
    
    def apply(self, input_signal): 
        output_signal = numpy.asarray(scipy.signal.lfilter(self.b, self.a, input_signal, axis=0), dtype=numpy.int16)
        return output_signal
    
class SignalCombiner(object):
    
    def __init__(self, list_of_weights0): 
        self.list_of_weights=list_of_weights0

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
                                    None if layer_index==0 else SignalCombiner([random.uniform(0,1) for i in xrange(num_units_per_layer)])
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
                    if layer_index==1 and unit_index==0:
                        print unit[SIGNAL_COMBINER_INDEX].list_of_weights
#                        print numpy.sum([weight*input_signal for input_signal, weight in zip(output_signals_network[layer_index-1],unit[SIGNAL_COMBINER_INDEX].list_of_weights)], axis=0)
#                        print numpy.sum([6*output_signals_network[layer_index-1][0],7*output_signals_network[layer_index-1][1]], axis=0)
#                        print output_signals_network[layer_index-1]
#                        print [output_signals_network[layer_index-1][0],output_signals_network[layer_index-1][1]]
                        def p(i):
                            print i
                            return i
                        [p(e) for e in zip(output_signals_network[layer_index-1],unit[SIGNAL_COMBINER_INDEX].list_of_weights)]
                        print "Code's sum"
                        print [weight*input_signal for input_signal, weight in zip(output_signals_network[layer_index-1],unit[SIGNAL_COMBINER_INDEX].list_of_weights)]
                        print "Our sum"
                        print [6.0*output_signals_network[layer_index-1][0],7.0*output_signals_network[layer_index-1][1]]
#                        exit()
                        print "Code's input"
                        print current_input_signal
                output_signals_network[layer_index][unit_index] = unit[FILTER_INDEX].apply(current_input_signal)
        print "Code's output"
        print output_signals_network[1][0]
        return self.final_combiner.apply(output_signals_network[-1])
    
