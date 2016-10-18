#!/usr/bin/python

"""

Data Driven Reverb Simulator

TODO:
    Update usage()
    Read http://www.eas.uccs.edu/~mwickert/ece2610/lecture_notes/ece2610_chap8.pdf on how to solve for coefficients

"""

import os
import sys
import pdb
import time
import scipy.io.wavfile
import scipy.signal
import numpy
from util import *
from FilterNetwork import *

START_TIME=time.time()

def convert_hertz(freq, sample_rate=44100.0):
    # convert frequency in hz to units of pi rad/sample
    return freq * 2.0 / sample_rate

def usage(): 
    # Sample Usage: python main.py input.wav ./results -denoising_sigma 2
    print >> sys.stderr, ''
    print >> sys.stderr, 'Usage: python '+__file__+' input_image out_dir <options>'
    print >> sys.stderr, ''
    print >> sys.stderr, 'Sample Usage: python '+__file__+' input.png ./results -denoising_sigma 2'
    print >> sys.stderr, ''
    print >> sys.stderr, 'EXPLANATION TEXT.'
    print >> sys.stderr, ''
    print >> sys.stderr, 'Options:'
    print >> sys.stderr, ''
    print >> sys.stderr, '    -denoising_sigma <float>'
    print >> sys.stderr, '        We first smooth the data with a Gaussian filter. The default value is 2.0.'
    print >> sys.stderr, ''
    exit(1)

def main():
    if len(sys.argv) < 3:
        usage()
    
    # Get Params
    input_wav_location = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2]) # the output directory
    makedirs(out_dir)
    denoising_sigma = float(get_command_line_param_val_default_value(sys.argv,'-denoising_sigma',2))
    
    print "Parameters: "
    print "    Input WAV File Location:",input_wav_location
    print "    Output Directory:",out_dir
    print "    Denoising Sigma:",denoising_sigma
    print 
    
    input_sample_rate, input_samples = scipy.io.wavfile.read(input_wav_location)
#    input_samples = numpy.asarray(input_samples, dtype=numpy.int16)
    output_wav_location = os.path.join(out_dir,"output.wav")
    
#    bn=1
#    b = numpy.ones(bn)/float(bn)
#    an=10000
#    a = numpy.zeros(an)
#    a[0]=1
#    a[an-1]=-0.5
#    
#    output_samples = numpy.asarray(scipy.signal.lfilter(b, a, input_samples, axis=0), dtype=numpy.int16)
#    
#    print input_samples[100:110]
#    print output_samples[100:110]
#    print output_samples[100:110]-input_samples[:10]
#    
#    scipy.io.wavfile.write(output_wav_location, input_sample_rate, output_samples)
#    
#    print 
#    print scipy.signal.lfilter(b, a, [1]+100*[0], axis=0)
    
    fn = FilterNetwork(num_layers=2, num_units_per_layer=2, num_fir_coefficients=3, num_iir_coefficients=3)
    
    fn.network[0][0][0].b = numpy.array([1,2,3], dtype='float')
    fn.network[0][0][0].a = numpy.array([4,5,6], dtype='float')
    
    fn.network[0][1][0].b = numpy.array([7,8,9], dtype='float')
    fn.network[0][1][0].a = numpy.array([10,11,12], dtype='float')
    
    fn.network[1][0][0].b = numpy.array([13,14,15], dtype='float')
    fn.network[1][0][0].a = numpy.array([16,17,18], dtype='float')
    fn.network[1][0][1].list_of_weights = numpy.array([6,7], dtype='float')
    
    fn.network[1][1][0].b = numpy.array([19,20,21], dtype='float')
    fn.network[1][1][0].a = numpy.array([22,23,24], dtype='float')
    fn.network[1][1][1].list_of_weights = numpy.array([8,9], dtype='float')
    
    fn.final_combiner.list_of_weights = [1,1]
    
    filt = lambda bb,aa,ss: numpy.asarray(scipy.signal.lfilter(bb, aa, ss, axis=0), dtype=numpy.int16)
    
    l0u0_b = fn.network[0][0][0].b
    l0u0_a = fn.network[0][0][0].a
    l0u1_b = fn.network[0][1][0].b
    l0u1_a = fn.network[0][1][0].a
    l1u0_b = fn.network[1][0][0].b
    l1u0_a = fn.network[1][0][0].a
    l1u1_b = fn.network[1][1][0].b
    l1u1_a = fn.network[1][1][0].a
    
    l0u0_output = filt(l0u0_b,l0u0_a,input_samples)
    l0u1_output = filt(l0u1_b,l0u1_a,input_samples)
    l1u0_output = filt(l0u0_b,l0u0_a,numpy.sum([6.0*l0u0_output,7.0*l0u1_output],axis=0))
#    l1u0_output = filt(l0u0_b,l0u0_a,numpy.sum([weight*input_signal for input_signal, weight in zip([l0u0_output,l0u1_output],[6,7])],axis=0))
    l1u1_output = filt(l0u1_b,l0u1_a,numpy.sum([8.0*l0u0_output,9.0*l0u1_output],axis=0))
    
    output_code = fn.apply(input_samples)
    output_manual = l1u0_output+l1u1_output
    
#    print input_samples
    print "Our input"
    print numpy.sum([6.0*l0u0_output,7.0*l0u1_output],axis=0)
    print "Our output"
    print l1u0_output
#    print output_code
#    print fn.final_combiner.list_of_weights
#    print fn.final_combiner.list_of_weights[0]*output_manual
    
    print 
    print 'Total Run Time: '+str(time.time()-START_TIME) 
    print 

if __name__ == '__main__': 
    main() 

