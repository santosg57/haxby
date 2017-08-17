#!/bin/python

from mvpa2.tutorial_suite import *

pati = '/home/inb/santosg/TUTO/tutorial_data/data/haxby2001'
mask_fname = os.path.join(pati, 'sub001', 'masks', 'orig', 'vt.nii.gz')

dhandle = OpenFMRIDataset(pati)
print dhandle.get_subj_ids()
print dhandle.get_task_descriptions()

print dir(dhandle)
print '-------------------------------'

task = 1   # object viewing task 
model = 1  # image stimulus category model 
subj = 1
run_datasets = []
for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
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
print dir(fds)
print len(fds)
print type(fds)
print fds.shape
print '-------------------------------------'
print fds.summary()

#-------------------------- Detrending ------------------

detrender = PolyDetrendMapper(polyord=1, chunks_attr='chunks')

detrended_fds = fds.get_mapped(detrender)
print detrended_fds.a.mapper

#--------------------------- Normalization -------------

#zscorer = ZScoreMapper(param_est=('targets', ['rest'])) 

zscore(detrended_fds, param_est=('targets', ['rest']))
fds = detrended_fds
print fds.a.mapper

fds = fds[fds.sa.targets != 'rest']
print fds.shape

#---------------------------- Computing Patterns Of Activation ------------------------

rnames = {0: 'even', 1: 'odd'}
fds.sa['runtype'] = [rnames[c % 2] for c in fds.sa.chunks]

print fds.sa.runtype
print len(fds.sa.runtype)

averager = mean_group_sample(['targets', 'runtype'])
print type(averager)
fds = fds.get_mapped(averager)
print fds.shape

print fds.sa.targets

print fds.sa.chunks

clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')

clf.train(fds)

predictions = clf.predict(fds.samples)

print np.mean(predictions == fds.sa.targets)

print fds.sa.runtype

ds_split1 = fds[fds.sa.runtype == 'odd']
len(ds_split1)
ds_split2 = fds[fds.sa.runtype == 'even']
len(ds_split2)

clf.set_postproc(BinaryFxNode(mean_mismatch_error, 'targets'))
clf.train(ds_split2)
err = clf(ds_split1)

print np.asscalar(err)

clf.train(ds_split1)
err = clf(ds_split2)
print np.asscalar(err)

#----------------------------- Cross-validation ---------------

# disable post-processing again 
clf.set_postproc(None)
# dataset generator 
hpart = HalfPartitioner(attr='runtype')
# complete cross-validation facility 
cv = CrossValidation(clf, hpart)

cv_results = cv(fds)
print np.mean(cv_results)

print len(cv_results)

print cv_results.samples





