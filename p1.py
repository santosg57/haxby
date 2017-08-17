# Tutorial Introduction to PyMVPA
from mvpa2.tutorial_suite import *
import numpy as np
from glob import glob


# Dataset basics and concepts

data = [[  1,  1, -1],
   [  2,  0,  0],
   [  3,  1,  1],
   [  4,  0, -1]]
   
ds = Dataset(data)
print ds.shape
print len(ds)
print ds.nfeatures
print ds.samples

one_d = [ 0, 1, 2, 3 ]
one_ds = Dataset(one_d)
print one_ds.shape

m_ds = Dataset(np.random.random((3, 4, 2, 3)))
print m_ds.shape
print m_ds.nfeatures

# Attributes
ds.sa['some_attr'] = [ 0., 1, 1, 3 ]
print ds.sa.keys()

print type(ds.sa['some_attr'])
print ds.sa['some_attr'].value

print ds.sa['some_attr'].unique

print ds.sa.some_attr

ds.sa['invalid'] = 4

ds.sa['invalid'] = [ 1, 2, 3, 4, 5, 6 ]

ds.sa['literal'] = ['one', 'two', 'three', 'four']
print sorted(ds.sa.keys())

for attr in ds.sa:
   print "%s: %s" % (attr, ds.sa[attr].value.dtype.name)

print ds.nfeatures
ds.fa['my_fav'] = [0, 1, 0]
ds.fa['responsible'] = ['me', 'you', 'nobody']
print sorted(ds.fa.keys())

# For the entire dataset

ds.a['pointless'] = glob("*")
print 'setup.py' in ds.a.pointless

# Slicing, resampling, feature selection

print ds.samples
print ds[::2].samples

mask = np.array([True, False, True, False])
print ds[mask].samples

print ds[[0, 2]].samples

print ds[:, [1,2]].samples

subds = ds[[0,1], [0,2]]
print subds.samples

print ds.samples[[0,1], [0,2]]

