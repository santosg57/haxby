from mvpa2.tutorial_suite import *

data_path='/misc/charcot/jhevia/azalea_pymvpa/data' + str(1) + '/Azalea2017/'
mask_fname = os.path.join(data_path,"/misc/arwen/azalea/PPI/TPJ_previous/MasksSubjects/PH101_Cor01_left.nii.gz")

dhandle = OpenFMRIDataset(data_path)

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
ds = vstack(run_datasets, a=0)

print ds.shape

# pre-process
poly_detrend(ds, polyord=1, chunks_attr='chunks')
zscore(ds, param_est=('targets', ['rest']))
ds = ds[ds.sa.targets != 'rest']
# average
run_averager = mean_group_sample(['targets', 'chunks'])
ds = ds.get_mapped(run_averager)
print ds.shape

clf = LinearCSVMC()
cvte = CrossValidation(clf, NFoldPartitioner(),
                        errorfx=lambda p, t: np.mean(p == t))
cv_results = cvte(ds)
print np.mean(cv_results)

print type(cv_results)
print cv_results.samples

print '--------------------------- We Need To Take A Closer Look --------------------'

cvte = CrossValidation(clf, NFoldPartitioner(),
                        errorfx=lambda p, t: np.mean(p == t),
                        enable_ca=['stats'])
cv_results = cvte(ds)

print cvte.ca.stats.as_string(description=True)

print cvte.ca.stats.matrix









