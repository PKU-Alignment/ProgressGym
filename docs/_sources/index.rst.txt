.. progress-gym documentation master file, created by
   sphinx-quickstart on Mon Jul  1 10:50:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ProgressGym: Dataset & Experimental Framework for Progress-Aware Alignment
==========================================================================
Frontier AI systems, including large language models (LLMs), hold increasing influence over the epistemology of human users. Such influence can reinforce prevailing societal values, potentially contributing to the lock-in of misguided moral beliefs and, consequently, the perpetuation of problematic moral practices on a broad scale. We introduce progress alignment as a technical solution to mitigate this imminent risk.

To empower research in progress alignment, we introduce ProgressGym, an experimental 
framework that codifies real-world progress alignment challenges into concrete benchmarks.
Specifically, we introduce three core challenges: tracking evolving values (PG-Follow),
preemptively anticipating moral progress (PG-Predict), and regulating the feedback loop be-
tween human and AI value shifts (PG-Coevolve).

Aside from example alignment algorithms (Lifelong RLHF, Lifelong DPO, Extrapolative RLHF, Extrapolative DPO),
we provide APIs for user-implemented examinees as well as judges, so that you can evaluate your own algorithm in terms of progress-aware alignment, on your custom tasks.

Check out the :doc:`usage` section for basic usage.

Contents
--------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   running
   usage
   documentation
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
