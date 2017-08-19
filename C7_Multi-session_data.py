from mvpa2.tutorial_suite import *

data_path = '/home/inb/santosg/TUTO/tutorial_data/data/haxby2001/'

mask_fname = os.path.join(data_path, 'sub001', 'masks', 'orig', 'vt.nii.gz')

dhandle = OpenFMRIDataset(data_path)

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

#49
rnames = {0: 'even', 1: 'odd'}
fds.sa['runtype'] = [rnames[c % 2] for c in fds.sa.chunks]


averager = mean_group_sample(['targets', 'runtype'])
print type(averager)

fds = fds.get_mapped(averager)
print fds.shape

print fds.sa.targets

print fds.sa.chunks


print "\n -------------- There and back again a Mapper tale -----------------\n "

print fds

print fds.a.mapper

orig_data = fds.a.mapper.reverse(fds.samples)
print orig_data.shape


