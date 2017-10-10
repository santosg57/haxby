#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# stems
# searchlight
# to compute a full cross-validation analysis in each spherical region of interest (ROI) in the brain.
# subsequent analysis with inferential statistics
# we are going to enable
# to entertain us while we are waiting
# One aspect is worth mentioning
# To precondition this data
# we are not interested in
# scrambled pictures
# as otherwise
# we no longer need
# the localizer mask
# which feature attribute
# per each fold
# we are stripping all attributes
# For the upcoming plots







"""
Searchlight on fMRI data
========================

.. index:: Searchlight

The original idea of a spatial searchlight algorithm stems from a paper by
:ref:`Kriegeskorte et al. (2006) <KGB06>`, and has subsequently been used in a
number of studies. The most common use for a searchlight is to compute a full
cross-validation analysis in each spherical region of interest (ROI) in the
brain. This analysis yields a map of (typically) classification accuracies that
are often interpreted or post-processed similar to a GLM statistics output map
(e.g. subsequent analysis with inferential statistics). In this example we look
at how this type of analysis can be conducted in PyMVPA.

As always, we first have to import PyMVPA.
"""

from mvpa2.suite import *
import Lee_Aza_V2
import numpy as np

"""As searchlight analyses are usually quite expensive in terms of computational
resources, we are going to enable some progress output to entertain us while
we are waiting."""

# enable debug output for searchlight call
if __debug__:
    debug.active += ["SLC"]

"""The next few calls load an fMRI dataset, while assigning associated class
targets and chunks (experiment runs) to each volume in the 4D timeseries.  One
aspect is worth mentioning. When loading the fMRI data with
:func:`~mvpa2.datasets.mri.fmri_dataset()` additional feature attributes can be
added, by providing a dictionary with names and source pairs to the `add_fa`
arguments. In this case we are loading a thresholded zstat-map of a category
selectivity contrast for voxels ventral temporal cortex."""

# data path
#datapath = os.path.join(mvpa2.cfg.get('location', 'tutorial data'), 'haxby2001')

datapath = os.path.join('/home/inb/santosg/TUTO/tutorial_data/data')
print datapath

print '111============================================================================'


#dataset = load_tutorial_data(
#        path=datapath,
#        roi='brain',
#        add_fa={'vt_thr_glm': os.path.join(datapath,'haxby2001', 'sub001', 'masks',
#                                                     'orig', 'vt.nii.gz')})

dataset = Lee_Aza_V2.Lee_Aza_V2(5)
dataset.fa={'vt_thr_glm': os.path.join("/misc/arwen/azalea/PPI/TPJ_previous/MasksSubjects/PH101_Cor01_left.nii.gz")}

print dir(dataset)
print dataset.shape
print dataset.fa
print '222======================'

"""The dataset is now loaded and contains all brain voxels as features, and all
volumes as samples. To precondition this data for the intended analysis we have
to perform a few preprocessing steps (please note that the data was already
motion-corrected). The first step is a chunk-wise (run-wise) removal of linear
trends, typically caused by the acquisition equipment."""

#poly_detrend(dataset, polyord=1, chunks_attr='chunks')

"""Now that the detrending is done, we can remove parts of the timeseries we
are not interested in. For this example we are only considering volumes acquired
during a stimulation block with images of houses and scrambled pictures, as well
as rest periods (for now). It is important to perform the detrending before
this selection, as otherwise the equal spacing of fMRI volumes is no longer
guaranteed."""

print '333============================================================================'

print 'dataset.sa.targets: ', dataset.sa.targets[:100]
print 'dataset.sa.targets len : ', len(dataset.sa.targets)

for l in dataset.sa.targets:
   if l in ['REST', 'COOPERADOR', 'NCOOPERADOR']:
      print 'bien'
   else :
      print 'mal'

#dataset = dataset[np.array([l in ['rest', 'house', 'scrambledpix']
dataset = dataset[np.array([l in ['REST', 'COOPERADOR', 'NCOOPERADOR']
                           for l in dataset.sa.targets], dtype='bool')]

print '333444444=========================='
print dataset.shape


