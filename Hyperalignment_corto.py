from mvpa2.suite import *

verbose.level = 2

verbose(1, "Loading data...")
filepath = os.path.join('/home/inb/santosg/TUTO/tutorial_data/data/haxby2001/',
                        'hyperalignment_tutorial_data.hdf5.gz')
ds_all = h5load(filepath)

print(type(ds_all))
print(len(ds_all))
xx= ds_all[2]

print(type(xx))


print xx.shape
print xx.sa
print xx.fa
print xx.a
print xx.nsamples
print xx.samples

print(summary(xx))





