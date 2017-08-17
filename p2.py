
# Load fMRI data

bold_fname = os.path.join(tutorial_data_path, 'haxby2001', 'sub001', 'BOLD', 'task001_run001', 'bold.nii.gz')
ds = fmri_dataset(bold_fname)

print len(ds)
print ds.nfeatures
print ds.shape

mask_fname = os.path.join(tutorial_data_path, 'haxby2001', 'sub001', 'masks', 'orig', 'vt.nii.gz')
ds = fmri_dataset(bold_fname, mask=mask_fname)
print len(ds)
print ds.nfeatures

print ds.sa.time_indices[:5]
print ds.sa.time_coords[:5]
print ds.fa.voxel_indices[:5]
print ds.a.voxel_eldim
print ds.a.voxel_dim
print 'imghdr' in ds.a
