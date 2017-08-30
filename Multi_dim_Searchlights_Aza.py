from mvpa2.tutorial_suite import *
from mvpa2.misc.plot import plot_err_line, plot_bars

data_path = '/home/inb/jhevia/pymvpa2/data3/Azalea2017/'
#data_path = '/home/inb/santosg/Azalea2017/Azalea2017/'

mask_fname = os.path.join(data_path, 'sub001', 'masks', 'orig', 'PH101Cor1TPJright.nii.gz')

print '\n', data_path, '\n'

dhandle = OpenFMRIDataset(data_path)

task = 1   # object viewing task
model = 1  # image stimulus category model
subj = 1
run_datasets = []

for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
        print '\n run_id: ', run_id
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
ds = vstack(run_datasets, a=0)


print '\n SHAPE: ', ds.shape

print '\n----------------------Event-related Pre-processing Is Not Event-related -------------------'

poly_detrend(ds, polyord=1, chunks_attr='chunks')

orig_ds = ds.copy()

print '\n--------------------- Design Specification ----------------------'

events = find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)

print '\n numero de eventos: ', len(events)
print '\n ------ eventos --------'

for e in events[:4]:
   print e

events = [ev for ev in events if ev['targets'] in ['POSITIVO', 'NEGATIVO']]
print '\n numero de eventos: ',  len(events)
print '\n --------- eventos ----------'

for e in events:
   print e

print '\n -------------------Response Modeling--------------------'

for ev in events:
   if (ev['duration'] == 1):
      ev['onset'] -= 2
      ev['duration'] = 4
   else:
      ev['onset'] -= 2
      ev['duration'] = 5
   print 'ev -----------------------', ev

print len(events)

evds = eventrelated_dataset(ds, events=events)

print '-------------- Entra a crossvalidation -----------------------'

cvte = CrossValidation(GNB(), NFoldPartitioner(),
                        postproc=mean_sample())
sl = Searchlight(cvte,
                  IndexQueryEngine(voxel_indices=Sphere(1),
                                   event_offsetidx=Sphere(2)),
                  postproc=mean_sample())
res = sl(evds)

#-------------------------------------------------------------------------------

ts = res.a.mapper.reverse1(1 - res.samples[0])
# need to put the time axis last for export to NIfTI
ts = np.rollaxis(ts, 0, 4)
ni = nb.Nifti1Image(ts, ds.a.imgaffine).to_filename('ersl_aza.nii')


