Implementing Your Own Experiment
================================

To run the benchmark for your own alignment algorithm and assess its temporal alignment
abilities, you have to implement your own algorithm as a subclass of :class:`benchmark.framework.ExamineeBase`. Implement it
in a new ``.py`` file in ``algorithm``, then refer to :doc:`running` for guidance on how to run the benchmark on the examinee you implemented. 

Feeling daunted? Dom't worry. Read through the following descriptions of :class:`benchmark.framework.ExamineeBase`, and you'll understand the overall workings of the examinee class in our framework.

.. autoclass:: benchmark.framework.ExamineeBase
    :members:
    :noindex:

We have implemented the four baseline examinees as described in :doc:`index` in ``./algorithm``. You can turn to those implementations
for reference for your own implementation.

After implementation, use our pre-implemented *dummy* challenge to verify and debug your implementation.
 
.. code-block:: python

    $ python run_benchmark.py \\
      --algorithms=your_algorithm \\
      --challenges=Dummy \\
      --output_filename=dummy_debugging_run \\

After that, use ``run_benchmark.py`` as described in :doc:`index` to start your experiment.

The :class:`benchmark.framework.JudgeBase` class is the base class for all *Judges* (i.e. benchmark tasks). You can refer to documentations
for this class to better understand the workings between *Judge* and *Examinee* . Of course, you can implemente your own *Judge* class as well.
Our implementations of the three major tasks are in ``./challenges``, for your reference. 

.. autoclass:: benchmark.framework.JudgeBase
    :members:
    :noindex:

Similarly, use *dummy_examinee* to verify your implementation.

.. code-block:: python

    $ python run_benchmark.py \\
      --algorithms=Dummy \\
      --challenges=your_task \\
      --output_filename=dummy_debugging_run \\