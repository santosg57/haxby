# Event-related Data Analysis

from mvpa2.tutorial_suite import *
ds = get_raw_haxby2001_data(roi=(36,38,39,40))


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

# for each of the example voxels
for i, v in enumerate(example_voxels):
     # get a slicing array matching just to current example voxel
     slicer = np.array([tuple(idx) == v for idx in ds.fa.voxel_indices])
     # perform the timeseries segmentation just for this voxel
     evds_detrend = eventrelated_dataset(orig_ds[:, slicer], events=events)
     # now plot the mean timeseries and standard error
     for j, t in enumerate(evds.uniquetargets):
         l = plot_err_line(evds_detrend[evds_detrend.sa.targets == t].samples,
                           fmt=t_col[j], linestyle=vx_lty[i])
         # label this plot for automatic legend generation
         l[0][0].set_label('Voxel %i: %s' % (i, t))
# y-axis caption
_ = pl.ylabel('Detrended signal')
# visualize zero-level
> _ = pl.axhline(linestyle='--', color='0.6')
# put automatic legend
_ = pl.legend()
_ = pl.xlim((0,12))


for i, v in enumerate(example_voxels):
     slicer = np.array([tuple(idx) == v for idx in ds.fa.voxel_indices])
     evds_norm = eventrelated_dataset(ds[:, slicer], events=events)
     for j, t in enumerate(evds.uniquetargets):
         l = plot_err_line(evds_norm[evds_norm.sa.targets == t].samples,
                           fmt=t_col[j], linestyle=vx_lty[i])
         l[0][0].set_label('Voxel %i: %s' % (i, t))
_ = pl.ylabel('Normalized signal')
_ = pl.axhline(linestyle='--', color='0.6')
_ = pl.xlim((0,12))


# L1 normalization of sensitivity maps per split to make them
# comparable
normed = sens.get_mapped(FxMapper(axis='features', fx=l1_normed))
smaps = evds.a.mapper[-1].reverse(normed)

for i, v in enumerate(example_voxels):
     slicer = np.array([tuple(idx) == v for idx in ds.fa.voxel_indices])
     smap = smaps.samples[:,:,slicer].squeeze()
     l = plot_err_line(smap, fmt='ko', linestyle=vx_lty[i], errtype='std')
_ = pl.xlim((0,12))
_ = pl.ylabel('Sensitivity')
_ = pl.axhline(linestyle='--', color='0.6')
_ = pl.xlabel('Peristimulus volumes')


