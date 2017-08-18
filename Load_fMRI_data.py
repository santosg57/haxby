# Load fMRI data

bold_fname = os.path.join(tutorial_data_path, 'haxby2001', 'sub001',
                              'BOLD', 'task001_run001', 'bold.nii.gz')
ds = fmri_dataset(bold_fname)
print len(ds)

print ds.nfeatures

print ds.shape

mask_fname = os.path.join(tutorial_data_path, 'haxby2001', 'sub001',
                           'masks', 'orig', 'vt.nii.gz')

ds = fmri_dataset(bold_fname, mask=mask_fname)
print len(ds)
print ds.nfeatures

print ds.sa.time_indices[:5]
print ds.sa.time_coords[:5]

print ds.fa.voxel_indices[:5]
print ds.a.voxel_eldim
print ds.a.voxel_dim
print 'imghdr' in ds.a

print print ds.a.mapper

stripped = ds.copy(deep=False, sa=['time_coords'], fa=[], a=[])
print stripped

import tempfile, shutil
# create a temporary directory
tempdir = tempfile.mkdtemp()
ds.save(os.path.join(tempdir, 'mydataset.hdf5'))


ds.save(os.path.join(tempdir, 'mydataset.gzipped.hdf5'), compression=9)
h5save(os.path.join(tempdir, 'mydataset.gzipped.hdf5'), ds, compression=9)


loaded = h5load(os.path.join(tempdir, 'mydataset.hdf5'))
print np.all(ds.samples == loaded.samples)

# cleanup the temporary directory, and everything it includes
shutil.rmtree(tempdir, ignore_errors=True)


