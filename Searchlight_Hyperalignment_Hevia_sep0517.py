from mvpa2.suite import *

verbose.level = 2

verbose(1, "Loading data...")

def Lee_Sujeto(direc):
    data_path = '/misc/charcot/jhevia/azalea_pymvpa/data'+str(direc)+'/Azalea2017/'

    #mask_fname = os.path.join(data_path, 'sub001', 'masks', 'orig', 'MFGleftBIN.nii.gz')
    mask_fname = os.path.join("/misc/arwen/azalea/PPI/antSFG/MasksSubjects/PH101_Cor01_left_BIN.nii.gz")
    print '\n', data_path, '\n'

    dhandle = OpenFMRIDataset(data_path)

    task = 1   # object viewing task
    model = 1  # image stimulus category model
    subj = 1
    run_datasets = []

    for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
        print '\n run_id, task, sujeto ', run_id, task, subj
        # load design info for this run
        run_events = dhandle.get_bold_run_model(model, subj, run_id)
        # load BOLD data for this run (with masking); add 0-based chunk ID
        run_ds = dhandle.get_bold_run_dataset(subj, task, run_id,
                                              chunks=run_id -1,
                                              mask=mask_fname)
        # convert event info into a sample attribute and assign as 'targets'
        run_ds.sa['targets'] = events2sample_attr(
                    run_events, run_ds.sa.time_coords, noinfolabel='REST')
        # additional time series preprocessing can go here
        run_datasets.append(run_ds)
#        print run_ds.sa.targets
        print '------------------------------------------------'
# this is PyMVPA's vstack() for merging samples from multiple datasets
# a=0 indicates that the dataset attributes of the first run should be used
# for the merged dataset
    ww = vstack(run_datasets, a=0)
    print ' shape ww'
    print ww.shape
    return ww

ds_all = []
for suj in range(33):
   print suj
   sujeto = suj+1
   if sujeto != 7 and sujeto !=9 and sujeto != 19 and sujeto != 34 and sujeto != 26 and sujeto != 30: 
      xx = Lee_Sujeto(sujeto)
      ds_all.append(xx)

#filepath = os.path.join('/home/inb/santosg/TUTO/tutorial_data/data/haxby2001/',
#                        'hyperalignment_tutorial_data.hdf5.gz')
#ds_all = h5load(filepath)

print 'type: \n'
print(type(ds_all))

print 'len: \n'
print(len(ds_all))

xx = ds_all[2]
print 'type: \n'

print type(xx)

print 'shape: \n'
print(xx.shape)

print 'sa: \n'
print xx.sa

print 'fa: \n'
print xx.fa

print 'a: \n'
print xx.a
print xx.nsamples

print 'resumen: \n'
print(summary(xx))

print '----------------------- zscore all datasets individually ----------------------'
#_ = [zscore(ds) for ds in ds_all]
#inject the subject ID into all datasets
for i, sd in enumerate(ds_all):
    print i, len(sd)
    sd.sa['subject'] = np.repeat(i, len(sd))
#number of subjects
nsubjs = len(ds_all)
#number of categories
ncats = len(ds_all[0].UT)
#number of run
nruns = len(ds_all[0].UC)
verbose(2, "%d subjects" % len(ds_all))
verbose(2, "Per-subject dataset: %i samples with %i features" % ds_all[0].shape)
verbose(2, "Stimulus categories: %s" % ', '.join(ds_all[0].UT))

print "-------------------------------------------------------------------------------"

# feature selection helpers
slhyper_start_time = time.time()
bsc_slhyper_results = []

clf = LinearCSVMC()

# same cross-validation over subjects as before
cv = CrossValidation(clf, NFoldPartitioner(attr='subject'),errorfx=mean_match_accuracy)


# leave-one-run-out for hyperalignment training
for test_run in range(nruns):
  ds_train = [sd[sd.sa.chunks != test_run, :] for sd in ds_all]
  ds_test = [sd[sd.sa.chunks == test_run, :] for sd in ds_all]

# Initializing Searchlight Hyperalignment with Sphere searchlights of 3 voxelradius.
# Using 40% features in each SL and spacing centers at 3-voxels distance.
  slhyper = SearchlightHyperalignment(radius=3, featsel=0.4, sparse_radius=3)

# Performing searchlight hyperalignment on training data.
# This step is similar to regular hyperalignment, calling the searchlight hyperalignment object with a list of datasets. Searchlight Hyperalignment returns a list of mappers corresponding to
# subjects in the same order as the list of datasets we passed in.
  slhypmaps = slhyper(ds_train)


# Applying hyperalignment parameters is similar to applying any mapper in PyMVPA. We apply the hyperalignment parameters by running the test dataset through the forward() function of the mapper.
  ds_hyper = [h.forward(sd) for h, sd in zip(slhypmaps, ds_test)]
# Running between-subject classification as before.
  ds_hyper = vstack(ds_hyper)
  zscore(ds_hyper, chunks_attr='subject')
  res_cv = cv(ds_hyper)
  bsc_slhyper_results.append(res_cv)
  
bsc_slhyper_results = hstack(bsc_slhyper_results)
verbose(2, "done in %.1f seconds" % (time.time() - slhyper_start_time,))


#Comparing the results
verbose(1, "Average classification accuracies:")
verbose(2, "between-subject (searchlight hyperaligned): %.2f +/-%.3f" \
% (np.mean(bsc_slhyper_results),
np.std(np.mean(bsc_slhyper_results, axis=1)) / np.sqrt(nsubjs - 1)))

