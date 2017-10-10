from mvpa2.tutorial_suite import *

def Lee_Aza():
   data_path = '/home/inb/jhevia/pymvpa2/data2/Azalea2017/'
   #data_path = '/home/inb/santosg/Azalea2017/Azalea2017/'
   mask_fname = os.path.join(data_path, 'sub001', 'masks', 'orig', 'PH101Cor1TPJright.nii.gz')

   dhandle = OpenFMRIDataset(data_path)

   task = 1   # object viewing task
   model = 1  # image stimulus category model
   subj = 1
   run_datasets = []

   for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
#        print '\n run_id: ', run_id
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
   fds = vstack(run_datasets, a=0)
   return fds



