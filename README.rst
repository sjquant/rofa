Rofa
=====

Introduction
------------

Interests in **Quant investing** (or trading) are growing fast every day. Many people try to create profitable strategies and test it using past data (It is called 'backetest' by quants). It is good phenomenon in that it can prevent people from wasting their money with unverified strategies, or just news) and enables investing with more statistical approaches. However, some people started to cling to just higher CAGR (compound annual growth rate) and total returns and ignore other statistical performance to ensure robustness of strategy. CAGR is just one factor to evaluate strategy, and if you don't care other evaluations, your investing can be in danger in the future. See below pictures.

.. image:: https://trello-attachments.s3.amazonaws.com/5cff44a05a8aa00f048b6f41/431x282/655dde558a4aa7fbd48f82059130b5fb/image.png

'Strategy B' has underperformed 'Strategy A' before 2014, but it outperformed 'Strategy A' at the end in the perspective of total returns and CAGR. But as you know, we cannot say 'Strategy B' is more robust than 'Strategy A'

**Rofa** is abbreviation for 'Robust Factor'. This module helps you to check factor's robustness. All you do is just prepare data with certain formats, and ask rofa to simulate, and evaluate, and make summarized graph of it. Now you had powerful quant tool **'Rofa'**.

Installation
------------

.. code::

    pip install rofa

.. note::

    You can use any editor to use python, but I recommend using jupyter notebook to start. jupyter notebook allows you to interactively run python code block by block. You can install jupyter notebook as follows.

.. code::

    pip install jupyter

To start it

.. code::

    jupyter notebook

Getting started
---------------

- **Import Rofa**

First of all, import ``rofa`` and ``QuantileSimulator`` from rofa

.. code:: python

    import rofa
    from rofa import QuantileSimulator

- **Registering daily returns data**

Unfortunately, rofa does not have any data with it. So, in order to run simulation, you need to ****register daily returns data** (pandas DataFrame). If you register returns data once, rofa will find returns data later without re-registering (You can change this option using ``save=False``).

.. code:: python

    import pandas as pd

    # Read Close Data
    returns= pd.read_pickle('../data/returns.pkl')  # Your returns data
    rofa.register_returns(returns)

.. note::

    returns dataframe data must have following format, where columns are asset symbols and index is date.

.. image:: https://trello-attachments.s3.amazonaws.com/5cff44a05a8aa00f048b6f41/950x265/e39cca22e2da9014b8bda5cfe78c6f40/image.png

- **Prepare data**

Data (pandas Dataframe) must have formats where columns are asset symbols (or code) and index is date such as returns. You can download data used in this example in here_.

.. _here: https://drive.google.com/drive/folders/1HnZYE0smawi_YoxcnTsdESEJuZDme2F5?usp=sharing

.. code:: python

    # Read Close Data
    close = pd.read_pickle('../data/close.pkl')  # Your data

.. image:: https://trello-attachments.s3.amazonaws.com/5cff44a05a8aa00f048b6f41/947x255/c8dd7064418c602ba01350f25be0a808/image.png

- **QuantileSimulator**

.. code:: python

    quan_sim = QuantileSimulator(close, rebalance_freq=10, bins=5)

In QuantileSimulator, first argument accepts factor data (close here). Additionaly, you can set ``rebalance_freq``, ``bins``, ``tax_rate``, ``weight_model`` and etc.

- **Run simulation**

Just run the simulation. Simulation logics are all done by ``rofa``

.. code:: python

    quan_sim.run()

- **Plot portfolio returns**

Simulation classes has plotter plugin inside it, which makes it possible to visuallize the simulation result.

.. code:: python

    quan_sim.plotter.plot_portfolio_returns()

.. image:: https://trello-attachments.s3.amazonaws.com/5cff44a05a8aa00f048b6f41/975x588/85eda2c8d19b247c944a86d95c0bc65d/image.png


From portfolio returns graph, we can compare overall performances and drawdowns of each portfolio.

- **Plot performance metrics (CAGR, MDD, Sharpe, Calmar)**

.. image:: https://trello-attachments.s3.amazonaws.com/5cff44a05a8aa00f048b6f41/1061x655/5ed868976346a3554c2677b6077ab1c5/image.png

From this graph, we can check performance metrics and check if there is strong relationship between factor and performance.

- **Plot rolling performance.**

.. image:: https://trello-attachments.s3.amazonaws.com/5cff44a05a8aa00f048b6f41/1076x656/a387b0aa8db6a379c9c578f986b42514/image.png

- **Wait, we can plot all at once**

You might have though about how come I can memorize all plot methods. Here's a method for you. ``plot_all`` plots all above. Super simple!

.. code:: python

    quan_sim.plotter.plot_all()

.. image:: https://trello-attachments.s3.amazonaws.com/5cff44a05a8aa00f048b6f41/476x897/58c2343f257c855bf8d6ec6b1bfd4a7c/image.png


TODO
----

- Add more performance indicators
- Optimize code for efficiency. There are some points to make code inefficient
- Create ``LongShortPlotter`` and make all methods used in ``QuantilePlotter``
- Add statistical analysis plugin such as ``Linear Regression``, ``t-test``, and ``ANOVA``
- Create ``NakedSimulator`` and add plotter plugins
- Create ``Evaluator Plugin`` Later
- Use ``numba`` or ``cython`` to improve performance
- Better documentation!