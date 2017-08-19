print "\n ----------------------- Classifiers ---------------------\n"

from mvpa2.tutorial_suite import *

data_path = '/home/inb/santosg/TUTO/tutorial_data/data/'

ds = get_haxby2001_data(data_path)

#---------------------------------------------------------------

clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')

clf.train(ds)

predictions = clf.predict(ds.samples)
print np.mean(predictions == ds.sa.targets)

print ds.sa.runtype

ds_split1 = ds[ds.sa.runtype == 'odd']
print len(ds_split1)

ds_split2 = ds[ds.sa.runtype == 'even']
print len(ds_split2)

clf.set_postproc(BinaryFxNode(mean_mismatch_error, 'targets'))
clf.train(ds_split2)
err = clf(ds_split1)
print np.asscalar(err)

clf.train(ds_split1)
err = clf(ds_split2)
print np.asscalar(err)

print "\n --------------------------- Cross-validation ----------------------\n"

# disable post-processing again
clf.set_postproc(None)
# dataset generator
hpart = HalfPartitioner(attr='runtype')
# complete cross-validation facility
cv = CrossValidation(clf, hpart)

cv_results = cv(ds)
np.mean(cv_results)

print len(cv_results)

print cv_results.samples

print "\n ------------------------- Any classifier, really ------------------- \n"

clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
cvte = CrossValidation(clf, HalfPartitioner(attr='runtype'))
cv_results = cvte(ds)
print np.mean(cv_results)


clf = LinearCSVMC()
cvte = CrossValidation(clf, HalfPartitioner(attr='runtype'))
cv_results = cvte(ds)
print np.mean(cv_results)

cvte = CrossValidation(clf, HalfPartitioner(attr='runtype'),
                        errorfx=lambda p, t: np.mean(p == t))
cv_results = cvte(ds)
print np.mean(cv_results)

# directory that contains the data files
datapath = os.path.join(tutorial_data_path, 'haxby2001')
# load the raw data
ds = load_tutorial_data(roi='vt')
# pre-process
poly_detrend(ds, polyord=1, chunks_attr='chunks')
zscore(ds, param_est=('targets', ['rest']))
ds = ds[ds.sa.targets != 'rest']
# average
run_averager = mean_group_sample(['targets', 'chunks'])
ds = ds.get_mapped(run_averager)
print ds.shape

print "\n ------------------------ We Need To Take A Closer Look ------------------- \n"

cvte = CrossValidation(clf, NFoldPartitioner(),
                        errorfx=lambda p, t: np.mean(p == t))
cv_results = cvte(ds)
print np.mean(cv_results)


print type(cv_results)
