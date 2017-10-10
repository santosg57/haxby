from mvpa2.tutorial_suite import *

tutorial_data_path = '/home/inb/santosg/TUTO/tutorial_data/data'

bold_fname = os.path.join(tutorial_data_path, 'haxby2001', 'sub001',
                           'BOLD', 'task001_run001', 'bold.nii.gz')
ds = fmri_dataset(bold_fname)

print len(ds)
print ds.nfeatures
print ds.shape



