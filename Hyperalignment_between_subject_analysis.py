from mvpa2.suite import *

verbose.level = 2

verbose(1, "Loading data...")
filepath = os.path.join('/home/inb/santosg/TUTO/tutorial_data/data/haxby2001/',
                        'hyperalignment_tutorial_data.hdf5.gz')
ds_all = h5load(filepath)

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

# zscore all datasets individually
_ = [zscore(ds) for ds in ds_all]
# inject the subject ID into all datasets
for i, sd in enumerate(ds_all):
    print i, len(sd)
    sd.sa['subject'] = np.repeat(i, len(sd))
# number of subjects
nsubjs = len(ds_all)
# number of categories
ncats = len(ds_all[0].UT)
# number of run
nruns = len(ds_all[0].UC)
verbose(2, "%d subjects" % len(ds_all))
verbose(2, "Per-subject dataset: %i samples with %i features" % ds_all[0].shape)
verbose(2, "Stimulus categories: %s" % ', '.join(ds_all[0].UT))



clf = LinearCSVMC()

# feature selection helpers
nf = 100
fselector = FixedNElementTailSelector(nf, tail='upper',
                                      mode='select', sort=False)
sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,
                                        enable_ca=['sensitivities'])
# create classifier with automatic feature selection
fsclf = FeatureSelectionClassifier(clf, sbfs)

print '-------------------------- Within-subject classification ------------------'

verbose(1, "Performing classification analyses...")
verbose(2, "within-subject...", cr=False, lf=False)
wsc_start_time = time.time()
cv = CrossValidation(fsclf,
                     NFoldPartitioner(attr='chunks'),
                     errorfx=mean_match_accuracy)
# store results in a sequence
wsc_results = [cv(sd) for sd in ds_all]
wsc_results = vstack(wsc_results)
verbose(2, " done in %.1f seconds" % (time.time() - wsc_start_time,))


print '---------------------------- Between-subject classification using anatomically aligned data ------------'

verbose(2, "between-subject (anatomically aligned)...", cr=False, lf=False)
ds_mni = vstack(ds_all)
mni_start_time = time.time()
cv = CrossValidation(fsclf,
                     NFoldPartitioner(attr='subject'),
                     errorfx=mean_match_accuracy)
bsc_mni_results = cv(ds_mni)
verbose(2, "done in %.1f seconds" % (time.time() - mni_start_time,))

print '---------------------------- Between-subject classification with Hyperalignment(TM) ------------------'

verbose(2, "between-subject (hyperaligned)...", cr=False, lf=False)
hyper_start_time = time.time()
bsc_hyper_results = []
# same cross-validation over subjects as before
cv = CrossValidation(clf, NFoldPartitioner(attr='subject'),
                     errorfx=mean_match_accuracy)

# leave-one-run-out for hyperalignment training
for test_run in range(nruns):
    # split in training and testing set
    ds_train = [sd[sd.sa.chunks != test_run, :] for sd in ds_all]
    ds_test = [sd[sd.sa.chunks == test_run, :] for sd in ds_all]

    # manual feature selection for every individual dataset in the list
    anova = OneWayAnova()
    fscores = [anova(sd) for sd in ds_train]
    featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
    ds_train_fs = [fs.forward(sd) for fs, sd in zip(featsels, ds_train)]


    # Perform hyperalignment on the training data with default parameters.
    # Computing hyperalignment parameters is as simple as calling the
    # hyperalignment object with a list of datasets. All datasets must have the
    # same number of samples and time-locked responses are assumed.
    # Hyperalignment returns a list of mappers corresponding to subjects in the
    # same order as the list of datasets we passed in.


    hyper = Hyperalignment()
    hypmaps = hyper(ds_train_fs)

    # Applying hyperalignment parameters is similar to applying any mapper in
    # PyMVPA. We start by selecting the voxels that we used to derive the
    # hyperalignment parameters. And then apply the hyperalignment parameters
    # by running the test dataset through the forward() function of the mapper.

    ds_test_fs = [fs.forward(sd) for fs, sd in zip(featsels, ds_test)]
    ds_hyper = [h.forward(sd) for h, sd in zip(hypmaps, ds_test_fs)]

    # Now, we have a list of datasets with feature correspondence in a common
    # space derived from the training data. Just as in the between-subject
    # analyses of anatomically aligned data we can stack them all up and run the
    # crossvalidation analysis.

    ds_hyper = vstack(ds_hyper)
    # zscore each subject individually after transformation for optimal
    # performance
    zscore(ds_hyper, chunks_attr='subject')
    res_cv = cv(ds_hyper)
    bsc_hyper_results.append(res_cv)

bsc_hyper_results = hstack(bsc_hyper_results)
verbose(2, "done in %.1f seconds" % (time.time() - hyper_start_time,))

print '-------------------------------- Comparing the results ---------------------------'

verbose(1, "Average classification accuracies:")
verbose(2, "within-subject: %.2f +/-%.3f"
        % (np.mean(wsc_results),
           np.std(wsc_results) / np.sqrt(nsubjs - 1)))
verbose(2, "between-subject (anatomically aligned): %.2f +/-%.3f"
        % (np.mean(bsc_mni_results),
           np.std(np.mean(bsc_mni_results, axis=1)) / np.sqrt(nsubjs - 1)))
verbose(2, "between-subject (hyperaligned): %.2f +/-%.3f" \
        % (np.mean(bsc_hyper_results),
           np.std(np.mean(bsc_hyper_results, axis=1)) / np.sqrt(nsubjs - 1)))








