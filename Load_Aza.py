from mvpa2.tutorial_suite import *
import Load_fMRI_data_Aza as aza

for sujeto in [1]: #range(1,5):
   if sujeto != 7 and sujeto !=9 and sujeto != 19 and sujeto != 34 and sujeto != 26 and sujeto != 30:
      ds = aza.Load_fMRI_data_mascara(sujeto)
      print len(ds)
      print ds.nfeatures
      print ds.shape

      print ds.sa.time_indices[:5]
      print ds.sa.time_coords[:5]
      print ds.fa.voxel_indices[:5]
      print ds.a.voxel_eldim
      print ds.a.voxel_dim
      print 'imghdr' in ds.a

      print ds.a.mapper
      stripped = ds.copy(deep=False, sa=['time_coords'], fa=[], a=[])
      print stripped






