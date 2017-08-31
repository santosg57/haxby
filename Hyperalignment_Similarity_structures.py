from mvpa2.suite import *

filepath = os.path.join('/home/inb/santosg/TUTO/tutorial_data/data/haxby2001/',
                        'hyperalignment_tutorial_data.hdf5.gz')
ds_all = h5load(filepath)

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


