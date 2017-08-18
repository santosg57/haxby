# Looking here and there – Searchlights


from mvpa2.tutorial_suite import *
ds = get_haxby2001_data(roi='vt')
print ds.shape


aov = OneWayAnova()
f = aov(ds)
print f


aov = OneWayAnova(
         postproc=FxMapper('features',
                           lambda x: x / x.max(),
                           attrfx=None))

f = aov(ds)
print f.samples.max()



# Searching, searching, searching, ...


clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
cv = CrossValidation(clf, HalfPartitioner())


sl = sphere_searchlight(cv, radius=3, postproc=mean_sample())


res = sl(ds)
print res


ds = get_haxby2001_data_alternative(roi=0)
print ds.nfeatures


res = sl(ds)


sphere_errors = res.samples[0]
res_mean = np.mean(res)
res_std = np.std(res)
# we deal with errors here, hence 1.0 minus
chance_level = 1.0 - (1.0 / len(ds.uniquetargets))


frac_lower = np.round(np.mean(sphere_errors < chance_level - 2 * res_std), 3)


