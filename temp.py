#!/usr/bin/env python


from mvpa2.suite import *
import Lee_Aza_V2

if __debug__:
    debug.active += ["SLC"]


#datapath = os.path.join('/home/inb/santosg/TUTO/tutorial_data/data')
#dataset = load_tutorial_data(
#        path=datapath,
#        roi='brain',
#        add_fa={'vt_thr_glm': os.path.join(datapath,'haxby2001', 'sub001', 'masks',
#                                                     'orig', 'vt.nii.gz')})

dataset = Lee_Aza_V2.Lee_Aza_V2(5)
dataset.fa={'vt_thr_glm': os.path.join("/misc/arwen/azalea/PPI/TPJ_previous/MasksSubjects/PH101_Cor01_left.nii.gz")} 

print 'C: ', dataset.C
#print 'O: ', dataset.O
#print 'S: ', dataset.S
print 'T: ', dataset.T
print 'UC: ', dataset.UC
print 'UT: ', dataset.UT
print 'a: ', dataset.a
print 'chunks: ', dataset.chunks
print 'fa: ', dataset.fa
#print 'item: ', dataset.item
print 'mapper: ', dataset.mapper
print 'nfeatures: ', dataset.nfeatures
#print 'nsamples: ', dataset.nsamples
print 'sa: ', dataset.sa
#print 'samples: ', dataset.samples
print 'shape: ', dataset.shape
print 'targets: ', dataset.targets[1:100]
print 'targets len: ', len(dataset.targets)

















