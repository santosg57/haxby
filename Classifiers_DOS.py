from mvpa2.tutorial_suite import *

pat = '/home/inb/santosg/TUTO/tutorial_data/data/'

# directory that contains the data files
datapath = os.path.join(pat)
# load the raw data
ds = load_tutorial_data(path=datapath,roi='vt')
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









