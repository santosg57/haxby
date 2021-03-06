from mvpa2.tutorial_suite import *

tutorial_data_path = '/home/inb/santosg/TUTO/tutorial_data/data'

bold_fname = os.path.join(tutorial_data_path, 'haxby2001', 'sub001',
                           'BOLD', 'task001_run001', 'bold.nii.gz')

mask_fname = os.path.join(tutorial_data_path, 'haxby2001', 'sub001',
                           'masks', 'orig', 'vt.nii.gz')

attr_fname = os.path.join(tutorial_data_path, 'haxby2001', 'sub001',
                           'BOLD', 'task001_run001', 'attributes.txt')
attr = SampleAttributes(attr_fname)

print len(attr.targets)
print np.unique(attr.targets)
print len(attr.chunks)
print np.unique(attr.chunks)

ds = fmri_dataset(samples=bold_fname,
                    targets=attr.targets, chunks=attr.chunks,
                    mask=mask_fname)

print ds.shape
print ds.sa




