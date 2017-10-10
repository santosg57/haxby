print '---------------------------------- Analysis setup --------------------------------'

from mvpa2.suite import *

verbose.level = 2

verbose(1, "Loading data...")

filepath = os.path.join('/home/inb/santosg/TUTO/tutorial_data/data/haxby2001/hyperalignment_tutorial_data.hdf5.gz')

ds_all = h5load(filepath)

print '\n---------------------------', type(ds_all)
#print '\n---------------shape-------', ds_all.shape
print '\n---------------len---------', len(ds_all)
#print '\n------------ds.nfeatures---', ds_all.nfeatures
#print '\n------------samples--------', bsc_slhyper_results.samples

# zscore all datasets individually
_ = [zscore(ds) for ds in ds_all]
# inject the subject ID into all datasets
for i, sd in enumerate(ds_all):
    print '\ni tipos', type(i), type(sd)
    sd.sa['subject'] = np.repeat(i, len(sd))
# number of subjects
nsubjs = len(ds_all)
# number of categories
ncats = len(ds_all[0].UT)

# number of run
nruns = len(ds_all[0].UC)

clf = LinearCSVMC()

print '---------------------------- Searchlight Hyperalignment ---------------------------'

verbose(1, "Performing classification analyses...")
verbose(2, "between-subject (searchlight hyperaligned)...", cr=False, lf=False)
# feature selection helpers
slhyper_start_time = time.time()
bsc_slhyper_results = []
# same cross-validation over subjects as before
cv = CrossValidation(clf, NFoldPartitioner(attr='subject'),
                     errorfx=mean_match_accuracy)

print '\n------------------nruns= ', nruns

# leave-one-run-out for hyperalignment training
for test_run in range(nruns):
    # split in training and testing set
    ds_train = [sd[sd.sa.chunks != test_run, :] for sd in ds_all]
    ds_test = [sd[sd.sa.chunks == test_run, :] for sd in ds_all]

    # Initializing Searchlight Hyperalignment with Sphere searchlights of 3 voxel radius.
    # Using 40% features in each SL and spacing centers at 3-voxels distance.
    slhyper = SearchlightHyperalignment(radius=3, featsel=0.4, sparse_radius=3)
    print '--------------- slhyper', type(slhyper)
    # Performing searchlight hyperalignment on training data.
    # This step is similar to regular hyperalignment, calling
    # the searchlight hyperalignment object with a list of datasets.
    # Searchlight Hyperalignment returns a list of mappers corresponding to
    # subjects in the same order as the list of datasets we passed in.
    slhypmaps = slhyper(ds_train)

    # Applying hyperalignment parameters is similar to applying any mapper in
    # PyMVPA. We apply the hyperalignment parameters by running the test dataset
    # through the forward() function of the mapper.
    ds_hyper = [h.forward(sd) for h, sd in zip(slhypmaps, ds_test)]

    # Running between-subject classification as before.
    ds_hyper = vstack(ds_hyper)
    zscore(ds_hyper, chunks_attr='subject')
    res_cv = cv(ds_hyper)
    bsc_slhyper_results.append(res_cv)

bsc_slhyper_results = hstack(bsc_slhyper_results)
verbose(2, "done in %.1f seconds" % (time.time() - slhyper_start_time,))

print '\n---------------------------', type(bsc_slhyper_results)
print '\n---------------shape-------', bsc_slhyper_results.shape
print '\n---------------len---------', len(bsc_slhyper_results)
print '\n------------ds.nfeatures---', bsc_slhyper_results.nfeatures
print '\n------------samples--------', bsc_slhyper_results.samples

