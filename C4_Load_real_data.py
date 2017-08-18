# Load real data

# directory that contains the data files
data_path = os.path.join(tutorial_data_path, 'haxby2001')
attr_fname = os.path.join(data_path, 'sub001',
                           'BOLD', 'task001_run001', 'attributes.txt')
attr = SampleAttributes(attr_fname)
print len(attr.targets)

print np.unique(attr.targets)

print len(attr.chunks)

print np.unique(attr.chunks)
[ 0.]


bold_fname = os.path.join(data_path,
                           'sub001', 'BOLD', 'task001_run001', 'bold.nii.gz')
mask_fname = os.path.join(tutorial_data_path, 'haxby2001',
                           'sub001', 'masks', 'orig', 'vt.nii.gz')
fds = fmri_dataset(samples=bold_fname,
                    targets=attr.targets, chunks=attr.chunks,
                    mask=mask_fname)
print fds.shape

print fds.sa



