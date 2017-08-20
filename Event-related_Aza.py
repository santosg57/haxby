from mvpa2.tutorial_suite import *
from mvpa2.misc.plot import plot_err_line, plot_bars

data_path = '/home/inb/jhevia/pymvpa2/data2/Azalea2017/'

mask_fname = os.path.join(data_path, 'sub001', 'masks', 'orig', 'PH101Cor1TPJright.nii.gz')

dhandle = OpenFMRIDataset(data_path)

task = 1   # object viewing task
model = 1  # image stimulus category model
subj = 1
run_datasets = []
for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
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
        print run_ds.sa.targets
        print '------------------------------------------------'
# this is PyMVPA's vstack() for merging samples from multiple datasets
# a=0 indicates that the dataset attributes of the first run should be used
# for the merged dataset
fds = vstack(run_datasets, a=0)


print 'SHAPE: ', fds.shape

print '----------------------Event-related Pre-processing Is Not Event-related -------------------'

poly_detrend(fds, polyord=1, chunks_attr='chunks')

orig_ds = fds.copy()

print '--------------------- Design Specification ----------------------'

events = find_events(targets=fds.sa.targets, chunks=fds.sa.chunks)

print len(events)
for e in events[:4]:
   print e

print '1111111111111111111111111111111111111111'

events = [ev for ev in events if ev['targets'] in ['COOPERADOR']]
print len(events)

for e in events[:4]:
   print e

print '-------------------Response Modeling--------------------'

# temporal distance between samples/volume is the volume repetition time
TR = np.median(np.diff(fds.sa.time_coords))

print 'TR: ', TR

# convert onsets and durations into timestamps
for ev in events:
   ev['onset'] = (ev['onset'] * TR)
   ev['duration'] = ev['duration'] * TR

evds = fit_event_hrf_model(fds,
   events,
   time_attr='time_coords',
   condition_attr=('targets', 'chunks'))

print 'LONGITUD DE evds: ', len(evds)

print '2222222222222222222222222222222'

zscore(evds, chunks_attr=None)

clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
cv = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
cv_glm = cv(evds)
print '%.2f' % np.mean(cv_glm)

print '333333333333333333333333333333333'

zscore(fds, param_est=('targets', ['REST']))
avgds = fds.get_mapped(mean_group_sample(['targets', 'chunks']))
avgds = avgds[np.array([t in ['COOPERADOR'] for t in avgds.sa.targets])]

cv_avg = cv(avgds)
print '%.2f' % np.mean(cv_avg)

print
print '----------------------- From Timeseries To Spatio-temporal Samples ----------------------'

print

zscore(fds, chunks_attr='chunks', param_est=('targets', 'rest'))

events = find_events(targets=fds.sa.targets, chunks=fds.sa.chunks)
events = [ev for ev in events if ev['targets'] in ['COOPERADOR']]

event_duration = 13

for ev in events:
   ev['onset'] -= 2
   ev['duration'] = event_duration

evds = eventrelated_dataset(fds, events=events)
print len(evds) == len(events)
print evds.nfeatures == fds.nfeatures * event_duration

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
   slicer = np.array([tuple(idx) == v for idx in fds.fa.voxel_indices])
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
_ = pl.axhline(linestyle='--', color='0.6')
# put automatic legend >>> _ = pl.legend() 
_ = pl.xlim((0,12))







