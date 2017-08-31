from mvpa2.suite import *

# get some example data
from mvpa2.testing.datasets import datasets
from mvpa2.misc.data_generators import random_affine_transformation
ds4l = datasets['uni4large']

print type(ds4l)
print ds4l.summary()

# generate a number of distorted variants of this data
dss = [random_affine_transformation(ds4l) for i in xrange(4)]

print '--------------- generate a number of distorted variants of this data -----------\n'
print type(dss)
print len(dss)

xx =  dss[2]
print '-------- summary --------'
print xx.summary()

ha = Hyperalignment()
ha.train(dss)
mappers = ha(dss)
len(mappers)

