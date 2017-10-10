from mvpa2.tutorial_suite import *

import Multi_session_data_Aza as aza

fds = aza.Multi_Session(1)

print fds.summary()

print '------------------------------ Detrending ---------------------------'

print len(fds)

detrender = PolyDetrendMapper(polyord=1, chunks_attr='chunks')

detrended_fds = fds.get_mapped(detrender)
print detrended_fds.a.mapper

print '------------------------------ Normalization ---------------------------'

zscorer = ZScoreMapper(param_est=('targets', ['rest']))

zscore(detrended_fds, param_est=('targets', ['rest']))
fds = detrended_fds
print fds.a.mapper

fds = fds[fds.sa.targets != 'rest']
print fds.shape

print '------------------------------ Computing Patterns Of Activationn ---------------------------'

rnames = {0: 'even', 1: 'odd'}
fds.sa['runtype'] = [rnames[c % 2] for c in fds.sa.chunks]

print 'len runtype:', len(fds.sa.runtype)

averager = mean_group_sample(['targets', 'runtype'])
print type(averager)

fds = fds.get_mapped(averager)
print fds.shape

print fds.sa.targets
print fds.sa.chunks


