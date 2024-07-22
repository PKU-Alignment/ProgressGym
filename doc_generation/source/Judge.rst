Judge
=====
In ProgressGym, we evaluate an *Examinee* (an algorithm) through *tasks*. In the code, those tasks are instantialized through classes called *Judges*, which handles queries from *Examinees* and updates the *human proxy models*.
This page documents the base Judge class as well as the three Judges we've implemented.

.. autoclass:: benchmark.framework.JudgeBase
    :members:

.. autoclass:: challenges.coevolve.CoevolveJudge
    :members:

.. autoclass:: challenges.follow.FollowJudge
    :members:

.. autoclass:: challenges.predict.PredictJudge
    :members:
