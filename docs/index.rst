Welcome to armscan_env!
=========================================

Welcome to the armscan_env library.

This project has been developed as a Master's thesis project at the Technical University of Munich, chair of Computer Aided Medical Procedures and Augmented Reality (CAMPAR).
The thesis was supervised by Prof. Dr. Nassir Navab, and part of the project was developed in collaboration with the Applied AI Initiative.

The project aims to provide a simulation environment for the training of reinforcement learning agents in the context of medical image analysis.
In particular, the project focuses on the task of navigating to the standard anatomical plane of the carpal tunnel in 3D labeled MRI scans of the hand.

The project is built on top of the OpenAI Gym framework and provides a custom environment that simulates the task of navigating to the carpal tunnel in 3D MRI scans.
The environment is designed to be modular and extensible, allowing for easy integration of new tasks.

In the `Introduction` section, we provide a brief overview of the problem statement and motivation behind the project, as well as an introduction to reinforcement learning.
We also discuss related work in the field of medical image analysis and reinforcement learning. Finally, we outline the contributions of this project.
The methodology section describes the lower level details of the project, including data preprocessing, the logic of the loss function used as reward signal, and the clustering algorithm used to recognize the landmarks in the MRI scans.
The environment section provides a detailed description of the custom OpenAI Gym environment developed for this project, including the observation space, action space, and reward function.
The experiments section presents the results of training reinforcement learning agents in the environment and evaluates their performance.
Finally, the conclusion section summarizes the key findings of the project and outlines potential future work.

Notebooks and code examples are provided in the `Notebook tutorials` section, and the project's documentation is available in the `API Reference` section.

See the project's repository_  for more information.

Contributions and feedback are welcome! Please check the `contributing to armscan_env` section for more information.

.. _repository: https://github.com/appliedAI-Initiative/armscan_env


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
