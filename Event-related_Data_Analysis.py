from mvpa2.tutorial_suite import *
pat = '/home/inb/santosg/TUTO/tutorial_data/data/'

ds = get_raw_haxby2001_data(pat, roi=(36,38,39,40))

print ds.shape

# Event-related Pre-processing Is Not Event-related

poly_detrend(ds, polyord=1, chunks_attr='chunks')

orig_ds = ds.copy()

# Design Specification

events = find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)

print len(events)

for e in events[:4]:
   print e


events = [ev for ev in events if ev['targets'] in ['house', 'face']]
print len(events)

for e in events[:4]:
   print e

# Response Modeling

# temporal distance between samples/volume is the volume repetition time
TR = np.median(np.diff(ds.sa.time_coords))

# convert onsets and durations into timestamps

for ev in events:
   ev['onset'] = (ev['onset'] * TR)
   ev['duration'] = ev['duration'] * TR


evds = fit_event_hrf_model(ds,
   events,
   time_attr='time_coords',
   condition_attr=('targets', 'chunks'))
print len(evds)

print evds.shape

clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
cv = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
cv_glm = cv(evds)
print '%.2f' % np.mean(cv_glm)

zscore(ds, param_est=('targets', ['rest']))
avgds = ds.get_mapped(mean_group_sample(['targets', 'chunks']))
avgds = avgds[np.array([t in ['face', 'house'] for t in avgds.sa.targets])]

cv_avg = cv(avgds)
print '%.2f' % np.mean(cv_avg)

sl = sphere_searchlight(cv_avg, radius=3, postproc=mean_sample())

res = sl(ds)



