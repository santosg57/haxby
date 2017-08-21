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
