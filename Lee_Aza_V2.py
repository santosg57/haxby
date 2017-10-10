from mvpa2.suite import *
from mvpa2.support.pylab import pl
from mvpa2.misc.data_generators import noisy_2d_fx
from mvpa2.mappers.svd import SVDMapper
from mvpa2.mappers.mdp_adaptor import ICAMapper, PCAMapper
from mvpa2 import cfg

def Lee_Aza_V2(num_suj):
# num_suj hasta 33
    def Lee_Sujeto(direc):
        data_path = '/misc/charcot/jhevia/azalea_pymvpa/data'+str(direc)+'/Azalea2017/'
        #mask_fname = os.path.join(data_path,"/misc/arwen/azalea/PPI/TPJ_previous/MasksSubjects/PH101_Cor01_left.nii.gz")
        #mask_fname = os.path.join("/misc/charcot/jhevia/azalea_pymvpa/mascara_bin-mask2.nii.gz")
        mask_fname = os.path.join("/misc/arwen/azalea/niigui/PH101_410/PH101_Cor01.feat/mask.nii.gz")
        print '\n', data_path, '\n'

        dhandle = OpenFMRIDataset(data_path)

        task = 1   # object viewing task
        model = 1  # image stimulus category model
        subj = 1
        run_datasets = []

        for run_id in dhandle.get_task_bold_run_ids(task)[subj]:
            print '\n run_id, task, sujeto ', run_id, task, subj
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
#        print run_ds.sa.targets

            print '------------------------------------------------'
# this is PyMVPA's vstack() for merging samples from multiple datasets
# a=0 indicates that the dataset attributes of the first run should be used
# for the merged dataset
        ww = vstack(run_datasets, a=0)
        print ' shape ww'
        print ww.shape
        return ww

    ds_all = []
    for suj in range(num_suj):
        print suj
        sujeto = suj+1
        if sujeto != 7 and sujeto !=9 and sujeto != 19 and sujeto != 34 and sujeto != 26 and sujeto != 30:
            xx = Lee_Sujeto(sujeto)
            poly_detrend(xx, polyord=1, chunks_attr='chunks')
            _ = [zscore(xx) for xx in ds_all]
#            xx = xx[xx.sa.targets != "REST"]
            ds_all.append(xx)
    datos = vstack(ds_all, a=0)
    return datos


