from mvpa2.tutorial_suite import *

def Load_fMRI_data(num):
   tutorial_data_path = '/misc/charcot/jhevia/azalea_pymvpa/data' + str(num) + '/Azalea2017/'

   bold_fname = os.path.join(tutorial_data_path, 'sub001',
                           'BOLD', 'task001_run001', 'bold.nii.gz')
   ds = fmri_dataset(bold_fname) 
   return ds

def Load_fMRI_data_mascara(num):
   data_path = '/misc/charcot/jhevia/azalea_pymvpa/data' + str(num) + '/Azalea2017/'
   mask_fname = os.path.join(data_path,"/misc/arwen/azalea/PPI/TPJ_previous/MasksSubjects/PH101_Cor01_left.nii.gz")
   bold_fname = os.path.join(data_path, 'sub001',
                           'BOLD', 'task001_run001', 'bold.nii.gz')
   
   ds = fmri_dataset(bold_fname, mask=mask_fname)
   return ds





