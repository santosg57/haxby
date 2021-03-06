# Multi-dimensional Searchlights

from mvpa2.tutorial_suite import *

data_path = '/home/inb/santosg/TUTO/tutorial_data/data/'

ds = get_raw_haxby2001_data(data_path, roi=(36,38,39,40))

poly_detrend(ds, polyord=1, chunks_attr='chunks')
zscore(ds, chunks_attr='chunks', param_est=('targets', 'rest'))
events = find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
events = [ev for ev in events if ev['targets'] in ['house', 'face']]
event_duration = 13
for ev in events:
     print 'ev --------------', ev
     ev['onset'] -= 2
     ev['duration'] = event_duration
     print 'ev ================-', ev
evds = eventrelated_dataset(ds, events=events)

#-------------------------------------------------------------------------------

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
ni = nb.Nifti1Image(ts, ds.a.imgaffine).to_filename('ersl.nii')


os.unlink('ersl.nii')
