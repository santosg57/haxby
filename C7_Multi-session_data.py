# Multi-session data

task = 1   # object viewing task
model = 1  # image stimulus category model
subj = 1
run_datasets = []
for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
     # load design info for this run
     run_events = dhandle.get_bold_run_model(model, subj, run_id)
     # load BOLD data for this run (with masking); add 0-based chunk ID
     run_ds = dhandle.get_bold_run_dataset(subj, task, run_id,
                                           chunks=run_id -1,
                                           mask=mask_fname)
     # convert event info into a sample attribute and assign as 'targets'
     run_ds.sa['targets'] = events2sample_attr(
                 run_events, run_ds.sa.time_coords, noinfolabel='rest')
     # additional time series preprocessing can go here
     run_datasets.append(run_ds)

# this is PyMVPA's vstack() for merging samples from multiple datasets
# a=0 indicates that the dataset attributes of the first run should be used
# for the merged dataset
fds = vstack(run_datasets, a=0)


print fds.summary()


# Basic preprocessing

detrender = PolyDetrendMapper(polyord=1, chunks_attr='chunks')

detrended_fds = fds.get_mapped(detrender)
print detrended_fds.a.mapper


# Normalization

zscorer = ZScoreMapper(param_est=('targets', ['rest']))


zscore(detrended_fds, param_est=('targets', ['rest']))
fds = detrended_fds
print fds.a.mapper

fds = fds[fds.sa.targets != 'rest']
print fds.shape


# Computing Patterns Of Activation


rnames = {0: 'even', 1: 'odd'}
fds.sa['runtype'] = [rnames[c % 2] for c in fds.sa.chunks]


averager = mean_group_sample(['targets', 'runtype'])
print type(averager)

fds = fds.get_mapped(averager)
print fds.shape

print fds.sa.targets

print fds.sa.chunks


# There and back again – a Mapper’s tale

print ds

print ds.a.mapper

>>> orig_data = ds.a.mapper.reverse(ds.samples)
>>> orig_data.shape
(5, 4, 3)


print subds
print subds.a.mapper
print subds.nfeatures

revtest = np.arange(subds.nfeatures) + 10
print revtest

rmapped = subds.a.mapper.reverse1(revtest)
print rmapped.shape

print rmapped

print fds.a.mapper

print fds.nfeatures

revtest = np.arange(100, 100 + fds.nfeatures)
rmapped = fds.a.mapper.reverse1(revtest)
print rmapped.shape


rmapped_partial = fds.a.mapper[:2].reverse1(revtest)
print (rmapped == rmapped_partial).all()


print ’imghdr' in fds.a

nimg = map2nifti(fds, revtest)

