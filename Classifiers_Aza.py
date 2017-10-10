from mvpa2.tutorial_suite import *

import Classifiers_FUN_Aza as aza

ds = aza.Classifiers_FUN(1)
print ds.shape

clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')

clf.train(ds)

print 'The classification performance'
predictions = clf.predict(ds.samples)
print np.mean(predictions == ds.sa.targets)

print ds.sa.runtype


ds_split1 = ds[ds.sa.runtype == 'odd']
print len(ds_split1)

ds_split2 = ds[ds.sa.runtype == 'even']
print len(ds_split2)

print 'hence lower values represent more accurate classification'

clf.set_postproc(BinaryFxNode(mean_mismatch_error, 'targets'))
clf.train(ds_split2)
err = clf(ds_split1)
print np.asscalar(err)

clf.train(ds_split1)
err = clf(ds_split2)
print np.asscalar(err)

print '---------------------------- Cross-validation ------------------------'

# disable post-processing again
clf.set_postproc(None)

# dataset generator
hpart = HalfPartitioner(attr='runtype')

# complete cross-validation facility
cv = CrossValidation(clf, hpart)

print 'return the results of all cross-validation folds.'
cv_results = cv(ds)
print np.mean(cv_results)

print len(cv_results)
print cv_results.samples

print '------------------------- Any classifier, really -------------------------'

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





