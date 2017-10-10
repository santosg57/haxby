from mvpa2.suite import *
from mvpa2.support.pylab import pl
from mvpa2.misc.data_generators import noisy_2d_fx
from mvpa2.mappers.svd import SVDMapper
from mvpa2.mappers.mdp_adaptor import ICAMapper, PCAMapper
from mvpa2 import cfg

verbose.level = 2

verbose(1, "Loading data...")

def Lee_Sujeto(direc):
    data_path = '/misc/charcot/jhevia/azalea_pymvpa/data'+str(direc)+'/Azalea2017/'
    mask_fname = os.path.join(data_path,"/misc/arwen/azalea/PPI/TPJ_previous/MasksSubjects/PH101_Cor01_left.nii.gz")
    #mask_fname = os.path.join("/misc/charcot/jhevia/azalea_pymvpa/mascara_bin-mask2.nii.gz")
    #mask_fname = os.path.join("/misc/arwen/azalea/niigui/PH101_410/PH101_Cor01.feat/mask.nii.gz")
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
for suj in range(5):
   print suj
   sujeto = suj+1
   if sujeto != 7 and sujeto !=9 and sujeto != 19 and sujeto != 34 and sujeto != 26 and sujeto != 30: 
      xx = Lee_Sujeto(sujeto)
      poly_detrend(xx, polyord=1, chunks_attr='chunks')
      _ = [zscore(xx) for xx in ds_all]
      xx = xx[xx.sa.targets != "REST"]
      ds_all.append(xx)

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

#print '----------------------- zscore all datasets individually ----------------------'
#_ = [zscore(ds) for ds in ds_all]
#inject the subject ID into all datasets
for i, sd in enumerate(ds_all):
    print i, len(sd)
    sd.sa['subject'] = np.repeat(i, len(sd))
#number of subjects
print type(sd.sa)
print len(sd.sa)
nsubjs = len(ds_all)
print "nsubjs = ", nsubjs
#number of categories
ncats = len(ds_all[0].UT)
print "ncats =", ncats
#number of run
nruns = len(ds_all[0].UC)
print "nruns=", nruns
#verbose(2, "%d subjects" % len(ds_all))
#verbose(2, "Per-subject dataset: %i samples with %i features" % ds_all[0].shape)
#verbose(2, "Stimulus categories: %s" % ', '.join(ds_all[0].UT))

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
# This step is similar to regular hyperalignment, calling the searchlight hyperalignment object with a list of datasets. Searchlight Hyperalignment returns a list of mappers corresponding to subjects in the same order as the list of datasets we passed in.
  slhypmaps = slhyper(ds_train)


# Applying hyperalignment parameters is similar to applying any mapper in PyMVPA. We apply the hyperalignment parameters by running the test dataset through the forward() function of the mapper.
  ds_hyper = [h.forward(sd) for h, sd in zip(slhypmaps, ds_test)]
# Running between-subject classification as before.
  ds_hyper = vstack(ds_hyper)
  #zscore(ds_hyper, chunks_attr='subject')
  res_cv = cv(ds_hyper)
  bsc_slhyper_results.append(res_cv)
  
bsc_slhyper_results = hstack(bsc_slhyper_results)
mappers = slhyper(bsc_slhyper_results)

verbose(2, "done in %.1f seconds" % (time.time() - slhyper_start_time,))


#Comparing the results
verbose(1, "Average classification accuracies:")
verbose(2, "between-subject (searchlight hyperaligned): %.2f +/-%.3f" \
        % (np.mean(bsc_slhyper_results),
           np.std(np.mean(bsc_slhyper_results, axis=1)) / np.sqrt(nsubjs - 1)))

#ds_fs = [sd[:, fselector(fscores)] for sd in ds_all]

revtest = np.arange(100, 100 + ds_all.nfeatures)
ts = bsc_slhyper_results.a.mapper.reverse(revtest)
ts = bsc_slhyper_results.a.mapper.reverse(1 - bsc_slhyper_results.samples[0])
ts = np.rollaxis(ts, 0, 4)
ni = nb.Nifti1Image(ts, xx.a.imgaffine).to_filename('ersl.nii')
os.unlink('ersl.nii')




#intended_label_order = [2, 2]
#labels = ds_all[0].UT
#labels = labels[intended_label_order]
#pl.figure(figsize=(6, 6))
#sm_orig = [np.corrcoef(
#               sd.get_mapped(
#                   mean_group_sample(['targets'])).samples)
#                       for sd in ds_all]
#sm_orig_mean = np.mean(sm_orig, axis=0)
#sm_anat = np.corrcoef(xx.get_mapped(mean_group_sample(['targets'])))
#sm_hyper_mean = np.mean(
#    [np.corrcoef(
#        sd.get_mapped(mean_group_sample(['targets'])).samples)
#     for sd in ds_hyper],
#    axis=0)
#sm_hyper = np.corrcoef(ds_hyper.get_mapped(mean_group_sample(['targets'])))
## plot all three similarity structures
#for i, sm_t in enumerate((
#    (sm_orig_mean, "Average within-subject\nsimilarity"),
#    (sm_anat, "Similarity of group average\ndata (anatomically aligned)"),
#    (sm_hyper_mean, "Average within-subject\nsimilarity (hyperaligned data)"),
#    (sm_hyper, "Similarity of group average\ndata (hyperaligned)"),
#    )):
#    sm, title = sm_t
#    # reorder matrix columns to match label order
#    sm = sm[intended_label_order][:, intended_label_order]
#    pl.subplot(2, 2, i + 1) 
#    pl.imshow(sm, vmin=-1.0, vmax=1.0, interpolation='nearest')
#    pl.colorbar(shrink=.4, ticks=[-1, 0, 1])
#    pl.title(title, size=12) 
#    ylim = pl.ylim()
#    pl.xticks(range(ncats), labels, size='small', stretch='ultra-condensed',
#              rotation=45)
#    pl.yticks(range(ncats), labels, size='small', stretch='ultra-condensed',
#              rotation=45)
#    pl.ylim(ylim)



