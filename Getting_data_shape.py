# Getting data in shape

from mvpa2.tutorial_suite import *
ds = dataset_wizard(np.ones((5, 12)))
print ds.shape

print 'mapper' in ds.a

ds = dataset_wizard(np.ones((5, 4, 3)))
print ds.shape

print 'mapper' in ds.a

print print ds.a.mapper

myfavs = [1, 2, 8, 10]
subds = ds[:, myfavs]
print subds.shape

print 'mapper' in subds.a

print subds.a.mapper

fwdtest = np.arange(12).reshape(4,3)
print fwdtest

fmapped = subds.a.mapper.forward1(fwdtest)
print fmapped.shape

print fmapped



