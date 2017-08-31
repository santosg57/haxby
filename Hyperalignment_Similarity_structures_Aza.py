from mvpa2.suite import *

verbose.level = 2

verbose(1, "Loading data...")

def Lee_Sujeto(sujeto):
    data_path = '/home/inb/jhevia/pymvpa2/data'+str(sujeto)+'/Azalea2017/'

    mask_fname = os.path.join(data_path, 'sub001', 'masks', 'orig', 'GFII.nii.gz')

    print '\n', data_path, '\n'

    dhandle = OpenFMRIDataset(data_path)

    task = 1   # object viewing task
    model = 1  # image stimulus category model
    subj = 1
    run_datasets = []

    for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
        #  print '\n run_id: ', run_id
        # load design info for this run
        run_events = dhandle.get_bold_run_model(model, subj, run_id)
        # load BOLD data for this run (with masking); add 0-based chunk ID
        run_ds = dhandle.get_bold_run_dataset(subj, task, run_id,
                                              chunks=run_id -1)
#                                              mask=mask_fname)
        # convert event info into a sample attribute and assign as 'targets'
        run_ds.sa['targets'] = events2sample_attr(
                    run_events, run_ds.sa.time_coords, noinfolabel='REST')
        # additional time series preprocessing can go here
        run_datasets.append(run_ds)
#        print run_ds.sa.targets
#        print '------------------------------------------------'
# this is PyMVPA's vstack() for merging samples from multiple datasets
# a=0 indicates that the dataset attributes of the first run should be used
# for the merged dataset
    ww = vstack(run_datasets, a=0)
    print ' shape ww'
    print ww.shape
    return ww

ds_all = [0, 0, 0, 0]
for suj in range(4):
   print suj
   xx = Lee_Sujeto(suj+1)
   ds_all[suj] = xx
   print xx.summary()
   print '=============================================================================='


#filepath = os.path.join('/home/inb/santosg/TUTO/tutorial_data/data/haxby2001/',
                        'hyperalignment_tutorial_data.hdf5.gz')
#ds_all = h5load(filepath)

ncats = len(ds_all[0].UT)

# feature selection helpers
nf = 100
fselector = FixedNElementTailSelector(nf, tail='upper',
                                      mode='select', sort=False)


# feature selection as above
anova = OneWayAnova()
fscores = [anova(sd) for sd in ds_all]
fscores = np.mean(np.asarray(vstack(fscores)), axis=0)
# apply to full datasets
ds_fs = [sd[:, fselector(fscores)] for sd in ds_all]
#run hyperalignment on full datasets
hyper = Hyperalignment()
mappers = hyper(ds_fs)
ds_hyper = [m.forward(ds_) for m, ds_ in zip(mappers, ds_fs)]
# similarity of original data samples
sm_orig = [np.corrcoef(
               sd.get_mapped(
                   mean_group_sample(['targets'])).samples)
                       for sd in ds_fs]
# mean across subjects
sm_orig_mean = np.mean(sm_orig, axis=0)
# same individual average but this time for hyperaligned data
sm_hyper_mean = np.mean(
    [np.corrcoef(
        sd.get_mapped(mean_group_sample(['targets'])).samples)
     for sd in ds_hyper],
    axis=0)
# similarity for averaged hyperaligned data
ds_hyper = vstack(ds_hyper)
sm_hyper = np.corrcoef(ds_hyper.get_mapped(mean_group_sample(['targets'])))
# similarity for averaged anatomically aligned data
ds_fs = vstack(ds_fs)
sm_anat = np.corrcoef(ds_fs.get_mapped(mean_group_sample(['targets'])))


# class labels should be in more meaningful order for visualization
# (human faces, animals faces, objects)
intended_label_order = [2, 4, 1, 5, 3, 0, 6]
labels = ds_all[0].UT
labels = labels[intended_label_order]

pl.figure(figsize=(6, 6))
# plot all three similarity structures
for i, sm_t in enumerate((
    (sm_orig_mean, "Average within-subject\nsimilarity"),
    (sm_anat, "Similarity of group average\ndata (anatomically aligned)"),
    (sm_hyper_mean, "Average within-subject\nsimilarity (hyperaligned data)"),
    (sm_hyper, "Similarity of group average\ndata (hyperaligned)"),
    )):
    sm, title = sm_t
    # reorder matrix columns to match label order
    sm = sm[intended_label_order][:, intended_label_order]
    pl.subplot(2, 2, i + 1)
    pl.imshow(sm, vmin=-1.0, vmax=1.0, interpolation='nearest')
    pl.colorbar(shrink=.4, ticks=[-1, 0, 1])
    pl.title(title, size=12)
    ylim = pl.ylim()
    pl.xticks(range(ncats), labels, size='small', stretch='ultra-condensed',
              rotation=45)
    pl.yticks(range(ncats), labels, size='small', stretch='ultra-condensed',
              rotation=45)
    pl.ylim(ylim)

pl.show()


