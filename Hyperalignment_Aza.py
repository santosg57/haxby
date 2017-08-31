from mvpa2.suite import *

verbose.level = 2

verbose(1, "Loading data...")

data_path = '/home/inb/jhevia/pymvpa2/data2/Azalea2017/'
dhandle = OpenFMRIDataset(data_path)

print type(dhandle)

#filepath = os.path.join('/home/inb/santosg/TUTO/tutorial_data/data/haxby2001/',
#                        'hyperalignment_tutorial_data.hdf5.gz')
#ds_all = h5load(filepath)

#print 'type: \n'
#print(type(ds_all))

#print 'len: \n'
#print(len(ds_all))

#xx = ds_all[2]
#print 'type: \n'
#print type(xx)

#print 'shape: \n'
#print(xx.shape)

#print 'sa: \n'
#print xx.sa

#print 'fa: \n'
#print xx.fa

#print 'a: \n'
#print xx.a
#print xx.nsamples

#print 'resumen: \n'
#print(summary(xx))





