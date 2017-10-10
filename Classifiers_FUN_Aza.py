from mvpa2.tutorial_suite import *

import Multi_session_data_Aza as aza

def Classifiers_FUN(num):
   fds = aza.Multi_Session(num)
   detrender = PolyDetrendMapper(polyord=1, chunks_attr='chunks')
   detrended_fds = fds.get_mapped(detrender)
   zscorer = ZScoreMapper(param_est=('targets', ['rest']))
   zscore(detrended_fds, param_est=('targets', ['rest']))
   fds = detrended_fds
   fds = fds[fds.sa.targets != 'rest']
   rnames = {0: 'even', 1: 'odd'}
   fds.sa['runtype'] = [rnames[c % 2] for c in fds.sa.chunks]
   averager = mean_group_sample(['targets', 'runtype'])
   fds = fds.get_mapped(averager)
   return fds



