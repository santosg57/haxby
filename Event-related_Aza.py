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


print '\n SHAPE: ', fds.shape

print '\n----------------------Event-related Pre-processing Is Not Event-related -------------------'

poly_detrend(fds, polyord=1, chunks_attr='chunks')

orig_ds = fds.copy()

print '\n--------------------- Design Specification ----------------------'

events = find_events(targets=fds.sa.targets, chunks=fds.sa.chunks)

print '\n numero de eventos: ', len(events)
print '\n ------ eventos --------'

for e in events[:4]:
   print e

events = [ev for ev in events if ev['targets'] in ['COOPERADOR']]
print '\n numero de eventos: ',  len(events)
print '\n --------- eventos ----------'
for e in events[:4]:
   print e

print '\n -------------------Response Modeling--------------------'

# temporal distance between samples/volume is the volume repetition time
TR = np.median(np.diff(fds.sa.time_coords))

print '\n TR: ', TR

# convert onsets and durations into timestamps
for ev in events:
   ev['onset'] = (ev['onset'] * TR)
   ev['duration'] = ev['duration'] * TR

evds = fit_event_hrf_model(fds,
   events,
   time_attr='time_coords',
   condition_attr=('targets', 'chunks'))

print 'LONGITUD DE evds: ', len(evds)

zscore(evds, chunks_attr=None)

clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
cv = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
cv_glm = cv(evds)
print '\n %.2f' % np.mean(cv_glm)

zscore(fds, param_est=('targets', ['REST']))
avgds = fds.get_mapped(mean_group_sample(['targets', 'chunks']))
avgds = avgds[np.array([t in ['COOPERADOR'] for t in avgds.sa.targets])]

cv_avg = cv(avgds)
print '\n %.2f' % np.mean(cv_avg)
