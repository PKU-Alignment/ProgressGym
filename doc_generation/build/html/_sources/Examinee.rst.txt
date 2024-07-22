Examinee
========
ProgressGym aims to evaluate how well algorithms align a model according to progressing human values.
 Those algorithms are instantialized here by the *Examinee* class. For each alignment algorithm,
 you have to (and we have already) implement a sub-class of the base *Examinee* Class. You should specify the extra
(from those stored in the base *Examinee* class) information it stores and, most importantly, how it interacts with the *Judge* class at each time step.
 Please refer to the *Judge* document page if you're unfamiliar with what *Judges* are.

This page documents the base Examinee class as well as the baseline examinees we've implemented, so as to give you a better understanding of how ProgressGym works.

.. autoclass:: benchmark.framework.ExamineeBase
    :members:

.. autoclass:: algorithms.extrapolative_dpo.ExtrapolativeDPOExaminee
    :members:

.. autoclass:: algorithms.extrapolative_rlhf.ExtrapolativeRLHFExaminee
    :members:

.. autoclass:: algorithms.lifelong_rlhf.LifelongRLHFExaminee
    :members:

.. autoclass:: algorithms.lifelong_dpo.LifelongDPOExaminee
    :members: