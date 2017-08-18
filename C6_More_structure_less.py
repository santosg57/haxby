# More structure, less duplication of work

dhandle = OpenFMRIDataset(data_path)
print dhandle.get_subj_ids()

print dhandle.get_task_descriptions()

model = 1
subj = 1
run = 1
events = dhandle.get_bold_run_model(model, subj, run)
for ev in events[:2]:
     print ev

targets = events2sample_attr(events, fds.sa.time_coords,
                              noinfolabel='rest', onset_shift=0.0)
print np.unique([attr.targets[i] == t for i, t in enumerate(targets)])

print np.unique(attr.targets)

print len(fds), len(targets)

task = 1
fds = dhandle.get_bold_run_dataset(subj, task, run, mask=mask_fname)
print fds


