# Event-related Data Analysis

from mvpa2.tutorial_suite import *

data_path = '/home/inb/santosg/TUTO/tutorial_data/data/'

ds = get_raw_haxby2001_data(data_path, roi=(36,38,39,40))


poly_detrend(ds, polyord=1, chunks_attr='chunks')
orig_ds = ds.copy()


events = find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
print len(events)

for e in events[:4]:
    print e


events = [ev for ev in events if ev['targets'] in ['house', 'face']]
print len(events)

for e in events[:4]:
    print e


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


zscore(evds, chunks_attr=None)


clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
cv = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
cv_glm = cv(evds)
print '%.2f' % np.mean(cv_glm)


zscore(ds, param_est=('targets', ['rest']))
avgds = ds.get_mapped(mean_group_sample(['targets', 'chunks']))
avgds = avgds[np.array([t in ['face', 'house'] for t in avgds.sa.targets])]


cv_avg = cv(avgds)
print '%.2f' % np.mean(cv_avg)

zscore(ds, chunks_attr='chunks', param_est=('targets', 'rest'))

events = find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
events = [ev for ev in events if ev['targets'] in ['house', 'face']]

event_duration = 13
for ev in events:
     ev['onset'] -= 2
     ev['duration'] = event_duration


evds = eventrelated_dataset(ds, events=events)
print len(evds) == len(events)

print evds.nfeatures == ds.nfeatures * event_duration

print evds.a.mapper[-2:]


sclf = SplitClassifier(LinearCSVMC(),
                        enable_ca=['stats'])
sensana = sclf.get_sensitivity_analyzer()
sens = sensana(evds)


example_voxels = [(28,25,25), (28,23,25)]

# linestyles and colors for plotting
vx_lty = ['-', '--']
t_col = ['b', 'r']
