#!/usr/bin/python

import scipy.signal
import numpy
import math
import copy
from util import *

FILTER_INDEX=0
SIGNAL_COMBINER_INDEX=1

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
    
    def get_clone(self):
        return copy.deepcopy(self)
    
class SignalCombiner(object):
    
    def __init__(self, list_of_weights0): 
        self.list_of_weights=numpy.array(list_of_weights0, dtype=numpy.float64)

    def apply(self, list_of_input_signals): 
        output_signal = numpy.sum([weight*input_signal for input_signal, weight in zip(list_of_input_signals, self.list_of_weights)], axis=0)
        return output_signal

class FilterGeneticAlgorithm(object):
    
    def __init__(self, population_size0=10, num_generations0=10, elite_percent0=0.8, num_fir_coefficients0=100, num_iir_coefficients0=100, training_set_directory='./training_set'): 
        self.population_size = population_size0 
        self.num_generations = num_generations0 
        self.elite_percent = elite_percent0 
        self.population = [Filter(a0=[random.uniform(-1.0,1.0) for ii in xrange(num_iir_coefficients0)], b0=[random.uniform(-1.0,1.0) for ii in xrange(num_fir_coefficients0)]) for i in xrange(population_size)] 
        self.num_fir_coefficients=num_fir_coefficients0
        self.num_iir_coefficients=num_iir_coefficients0
        list_of_wav_file_names = [e for e in os.listdir(training_set_directory) if '.wav'==e[-4:]] 
        # our files will have names "in_[index]" and "out_[index]" 
        self.training_data = [] 
        for input_wav_file_name in [e for e in list_of_wav_file_names if 'in_'==e[:3]]:
            output_wav_file_name = input_wav_file_name.replace('in_','out_')
            if output_wav_file_name not in list_of_wav_file_names:
                break
            input_sample_rate, input_samples = scipy.io.wavfile.read(input_wav_location)
            output_sample_rate, output_samples = scipy.io.wavfile.read(output_wav_location)
            assertion(output_sample_rate==input_sample_rate, "Sample rates of training set datum input and output are not the same", error_code=1):
            self.training_data.append((input_samples, output_samples, output_sample_rate))
    
    def get_score(filt, training_sample_index):
        training_sample_input, training_sample_output, sample_rate = self.training_data[training_sample_index]
        filt_output = filt.apply(training_sample_input)
        return get_signal_similarity(training_sample_output, fn_output, sample_rate)
    
    def mutate(filt0):
        # Returns a new mutated filter
        # 50% chance it changes IIR or FIR coefficients
        # Changes a random number of consecutive coefficients 
        filt = fn0.get_clone()
        if random.randint(0,1)==0:
            start_index = random.randint(0,len(filt.a)-1)
            end_index = random.randint(start_index,len(filt.a)-1)
            filt.a[start_index:end_index]=[random.unform(-1.0,1.0) for i in xrange(end_index-start_index+1)]
        else: 
            start_index = random.randint(0,len(filt.b)-1)
            end_index = random.randint(start_index,len(filt.b)-1)
            filt.b[start_index:end_index]=[random.unform(-1.0,1.0) for i in xrange(end_index-start_index+1)]
        return filt
    
    def crossover(filt1,filt2):
        # Creates a new filter with a 50% probability of coefficient coming from either parent
        num_iir_coefficients0 = len(filt1.a)
        num_fir_coefficients0 = len(filt1.b)
        child = filt1.clone()
        for i in xrange(num_iir_coefficients0):
            parent = filt1 if random.randint(0,1)==0 else filt2
            child.a[i] = parent.a[i]
        for i in xrange(num_fir_coefficients0):
            parent = filt1 if random.randint(0,1)==0 else filt2
            child.b[i] = parent.b[i]
        return child
    
    def run_generation(self, num_generations_to_run=1):
        for generation_index in xrange(num_generations_to_run):
            training_sample_index = random.randint(0,len(self.training_data)-1)
            mutations = [mutate(p) for p in self.population]
            crossovers = [crossover(random.choice(self.population),random.choice(self.population)) for i in xrange(len(self.population))]
            population_and_scores = [(p,get_score(p,training_sample_index)) for p in self.population+mutations+crossovers]
            population_and_scores=sorted(population_and_scores,key=lambda x:-x[1])[:self.population_size]
            self.population=[ps[0] for ps in population_and_scores]

class FilterNetwork(object):
    
    def __init__(self, num_layers=3, num_units_per_layer=3, num_fir_coefficients=10000, num_iir_coefficients=10000): 
        # To access a filter unit, we use self.network[layer_index, unit_index]
        self.network = \
                        [
                            [
                                (
                                    Filter([random.uniform(-1.0,1.0) for i in xrange(num_iir_coefficients)],[random.uniform(-1.0,1.0) for i in xrange(num_fir_coefficients)]), 
                                    None if layer_index==0 else SignalCombiner([random.uniform(0.0,1.0) for i in xrange(num_units_per_layer)])
                                )
                                for unit_index in xrange(num_units_per_layer)
                            ] 
                            for layer_index in xrange(num_layers)
                        ]
        self.final_combiner = SignalCombiner([random.uniform(0,1) for i in xrange(num_units_per_layer)])
    
    def get_clone(self):
        return copy.deepcopy(self)
    
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
    
class FilterNetworkGeneticAlgorithm(object):
    
    def __init__(self, population_size0=10, num_generations0=10, elite_percent0=0.8, num_layers0=3, num_units_per_layer0=3, num_fir_coefficients0=100, num_iir_coefficients0=100, training_set_directory='./training_set'): 
        self.population_size = population_size0 
        self.num_generations = num_generations0 
        self.elite_percent = elite_percent0 
        self.population = [FilterNetwork(num_layers=num_layers0, num_units_per_layer=num_units_per_layer0, num_fir_coefficients=num_fir_coefficients0, num_iir_coefficients=num_iir_coefficients0) for i in xrange(population_size)] 
        list_of_wav_file_names = [e for e in os.listdir(training_set_directory) if '.wav'==e[-4:]] 
        # our files will have names "in_[index]" and "out_[index]" 
        self.training_data = [] 
        for input_wav_file_name in [e for e in list_of_wav_file_names if 'in_'==e[:3]]:
            output_wav_file_name = input_wav_file_name.replace('in_','out_')
            if output_wav_file_name not in list_of_wav_file_names:
                break
            input_sample_rate, input_samples = scipy.io.wavfile.read(input_wav_location)
            output_sample_rate, output_samples = scipy.io.wavfile.read(output_wav_location)
            assertion(output_sample_rate==input_sample_rate, "Sample rates of training set datum input and output are not the same", error_code=1):
            self.training_data.append((input_samples,output_samples, output_sample_rate))
    
    def get_score(fn, training_sample_index):
        training_sample_input, training_sample_output, sample_rate = self.training_data[training_sample_index]
        fn_output = fn.apply(training_sample_input)
        return get_signal_similarity(training_sample_output, fn_output, sample_rate)
    
    def mutate_combiner(fn0):
        # Returns a new mutated filter network
        # Picks a random unit
        # Changes a random number of weights
        fn = fn0.get_clone()
        num_layers = fn.get_num_layers()
        num_units_per_layer = fn.get_num_units_per_layer()
        layer_index = random.randint(0,num_layers-1)
        unit_index = random.randint(0,num_units_per_layer-1)
        unit = fn.network[layer_index][unit_index]
        # Mutate Signal Combiner
        num_weights = len(unit[SIGNAL_COMBINER_INDEX].list_of_weights)
        if unit[SIGNAL_COMBINER_INDEX] is not None:
            num_weights_to_modify = random.randint(1,num_weights)
            for i in xrange(num_weights_to_modify):
                unit[SIGNAL_COMBINER_INDEX].list_of_weights[random.randint(0,num_weights-1)]=random.uniform(0.0,1.0) # this doesn't guarantee that we mutate exactly num_weights_to_modify weights, but it is much less memory intensive
        return fn
    
    def mutate_IIR(fn0):
        # Returns a new mutated filter network
        # Picks a random unit
        # Changes a random number of IIR coefficients
        fn = fn0.get_clone()
        num_layers = fn.get_num_layers()
        num_units_per_layer = fn.get_num_units_per_layer()
        layer_index = random.randint(0,num_layers-1)
        unit_index = random.randint(0,num_units_per_layer-1)
        unit = fn.network[layer_index][unit_index]
        # Mutate IIR
        num_iir_coefficients = len(unit[FILTER_INDEX].a)
        num_iir_coefficients_to_modify = random.randint(1,num_iir_coefficients)
        for i in xrange(num_iir_coefficients_to_modify):
            unit[FILTER_INDEX].a[random.randint(0,num_iir_coefficients-1)]=random.uniform(-1.0,1.0) # this doesn't guarantee that we mutate exactly num_iir_coefficients_to_modify coefficients, but it is much less memory intensive
        return fn
    
    def mutate_FIR(fn0):
        # Returns a new mutated filter network
        # Picks a random unit
        # Changes a random number of FIR coefficients.
        fn = fn0.get_clone()
        num_layers = fn.get_num_layers()
        num_units_per_layer = fn.get_num_units_per_layer()
        layer_index = random.randint(0,num_layers-1)
        unit_index = random.randint(0,num_units_per_layer-1)
        unit = fn.network[layer_index][unit_index]
        # Mutate IIR
        num_fir_coefficients = len(unit[FILTER_INDEX].b)
        num_fir_coefficients_to_modify = random.randint(1,num_fir_coefficients)
        for i in xrange(num_fir_coefficients_to_modify):
            unit[FILTER_INDEX].b[random.randint(0,num_fir_coefficients-1)]=random.uniform(-1.0,1.0) # this doesn't guarantee that we mutate exactly num_fir_coefficients_to_modify coefficients, but it is much less memory intensive
        return fn
    
    def crossover(fn1,fn2):
        # Creates a new filter network with a 50% probability of each unit coming from either parent
        num_layers0 = fn1.get_num_layers()
        num_units_per_layer0 = fn1.get_num_units_per_layer()
        num_iir_coefficients0 = len(fn1.network[0][0][FILTER_INDEX].a)
        num_fir_coefficients0 = len(fn1.network[0][0][FILTER_INDEX].b)
        child = FilterNetwork(num_layers=num_layers0, num_units_per_layer=num_units_per_layer0, num_fir_coefficients=num_fir_coefficients0, num_iir_coefficients=num_iir_coefficients0)
        for layer_index in xrange(num_layers0):
            for unit_index in xrange(num_units_per_layer0):
                parent = fn1 if random.randint(0,1)==0 else fn2
                child.network[layer_index][unit_index][SIGNAL_COMBINER_INDEX].list_of_weights = numpy.array(parent.network[layer_index][unit_index][SIGNAL_COMBINER_INDEX].list_of_weights, dtype=numpy.float64)
                child.network[layer_index][unit_index][FILTER_INDEX] = Filter(parent.network[layer_index][unit_index][FILTER_INDEX])
        return child
    
    
    def run_generation(self, num_generations_to_run=1):
        for generation_index in xrange(num_generations_to_run):
            training_sample_index = random.randint(0,len(self.training_data)-1)
            population_and_scores = [(fn,get_score(fn,training_sample_index)) for fn in self.population]
            # FIR Mutations
            fir_mutations = [mutate_FIR(fn) for fn in self.population]
            fir_mutations_and_scores = [(mutated_fn,get_score(mutated_fn,training_sample_index)) for mutated_fn in fir_mutations]
            population_and_scores = sorted(population_and_scores+fir_mutations_and_scores, key=key=lambda x:-x[1])[:self.population_size]
            # IIR Mutations
            iir_mutations = [mutate_IIR(fn_score_pair[0]) for fn_score_pair in population_and_scores]
            iir_mutations_and_scores = [(mutated_fn,get_score(mutated_fn,training_sample_index)) for mutated_fn in iir_mutations]
            population_and_scores = sorted(population_and_scores+iir_mutations_and_scores, key=key=lambda x:-x[1])[:self.population_size]
            # Combiner Mutations
            combiner_mutations = [mutate_combiner(fn_score_pair[0]) for fn_score_pair in population_and_scores]
            combiner_mutations_and_scores = [(mutated_fn,get_score(mutated_fn,training_sample_index)) for mutated_fn in combiner_mutations]
            population_and_scores = sorted(population_and_scores+combiner_mutations_and_scores, key=key=lambda x:-x[1])[:self.population_size]
            # Cross Over
            children = [crossover(random.choice(population_and_scores)[0],random.choice(population_and_scores)[0]) for i in xrange(self.population_size)]
            children_and_scores = [(child_fn,get_score(child_fn,training_sample_index)) for child_fn in children]
            population_and_scores = sorted(population_and_scores+children_and_scores, key=key=lambda x:-x[1])[:self.population_size]
        self.population = [fn_score_pair[0] for fn_score_pair in population_and_scores]

