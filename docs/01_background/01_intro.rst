Introduction
============

Medical Motivation
------------------

This project developed from the desire to assist doctors in assessing peripheral nerve enlargement in patients affected by leprosy. Leprosy is the most common and aggressive infective cause of nerve enlargement which according to the World Health Organization (WHO) it is still present in more than 120 countries, infecting more than 200,000 new victims reported every year :footcite:`WHO.2023`            . Also known as Hansen’s disease, leprosy is caused by the bacterium *Mycrobacterium leprae*. Since this bacterium develops best in regions of the body with cooler temperatures, it usually develops best in superficial areas, attacking accessible nerves where the temperatures are lower :footcite:`White.2015`            . Detecting the enlargement of accessible nerves is crucial in assessing patients with peripheral nerve disorders and monitoring their response to treatment.

There are different methods to asses nerve enlargement, such as palpation, motor and sensory testing, nerve conduction studies, and imaging techniques. Among the imaging techniques, ultrasound (US) is usually preferred because it is non-invasive, portable, real-time, and cost-effective :footcite:`Bathala.2017`            .

Because leprosy is mainly spread in developing countries and rural areas, it might often be difficult to diagnose or monitor the disease. Some of the reasons are its different clinical presentations, as well as the lack of medical experienced personnel, social stigma, and forced isolation :footcite:`White.2015`            .

Automatic solutions can be of great support to tackle these problems. AI-enabled diagnostic assistants can help to identify suspected symptoms and make up for the deficiency of expert personnel. Moreover, robotic solutions can address the unwillingness of human operators to work in close contact with the patients, be the reason the social dogma, the forced isolation, or the fear of contagion.

Hansen’s disease is however not common in developed countries, where it was eliminated as a public health problem in 2000 :footcite:`WHO.2023`            . A prevalent reason for peripheral nerve enlargement in the world is nerve entrapment syndromes. In particular, carpal tunnel syndrome (CTS) is the most common nerve entrapment injury, affecting around 10 million people annually :footcite:`Silvestri.2018`            . Symptoms are pain and even paresthesia of the thumb, index digit, and long digit.

Sonographic imaging is the preferred technique for a quick diagnosis of CTS, however, it also has some important limitations. One of the main drawbacks of US scanning is the need for standardization to allow repeatable scans for recurrent evaluations and monitoring. Moreover, it requires skilled operators, and operator training requires a large amount of time, which represents a problem where there is a lack of experienced sonographers. In addition, ultrasound imaging is strongly operator-dependent, since skills and experience are crucial for proper examination, and therefore it is very sensitive in terms of diagnostic accuracy. Nerve segmentation in particular is very challenging because the low contrast of nerve tissue and its low image variability from other tissues makes segmentation difficult. Lastly, the repetitive task can lead sonographers to develop musculoskeletal disorders and regional pain :footcite:`Evans.2009,Roll.2014`            .

Robotic US Systems (RUSS) might offer a solution to the problems discussed above using teleoperated, semi-autonomous, or fully autonomous systems. Furthermore, RUSS unlocks the possibility to offer assistance where there is a lack of medical expertise, as well as to separate the operator from the patient when there is a potential risk of contagion.

For the purpose of fully autonomous scans, it is necessary to develop systems that are able to navigate to the standard US planes at which determined structures can be recognized. Such employments are demanding because of the high dimensionality of the state space (current US image view, probe position, contact force, reference to human body) that must be correctly interpreted. Navigating to a goal state based on a series of observations is particularly well-suited for methods in Reinforcement Learning (RL).

Several works have developed simulated environments to train RL agents for navigation tasks. All these works have however something in common: they use a reward function based on the Euclidean distance of the current probe pose from the goal position. This means that the goal position must be labeled and it is used as a reference during training. The limit of such approaches is that the optimal position is not always known, and it is difficult to demonstrate a proper generalization of the learned policy when scanning different anatomies or even different patients.

Moreover, the datasets used to train such solutions are usually 3D volumes of ultrasound scans gathered from a small set of participants. Training an agent on an ultrasound image-based environment is however a difficult task because of the limited open source availability, to total absence, of US 3D volumes, as well as the difficulty of collecting new clinical data. It requires a large amount of high-quality labeled real US data, which is a time-consuming and labor-intensive acquisition. Lastly, the absence of probe position logging when registering a scan, and the lack of standardization techniques make it difficult to reuse ultrasound videos of scans performed by expert practitioners.

Thesis objectives
-----------------

The vision behind this work is to learn automatic image navigation to a standard plane, such as the carpal tunnel, to enable robot autonomous US scanning. This would offer great evaluation assistance to doctors, making the procedure less dependent on the operators and decreasing the evaluation variability. Moreover, it would permit to perform repeatable scans for recurrent monitoring, and it could reduce the time of a medical assessment as well as that of medical training.

The primary objective of this thesis is to develop a navigation system that relies solely on real-time image observations, defining an observable reward that can be assessed from the current observation. This observable reward enhances the predictability and robustness of the agent. The proposed method for autonomous navigation in a 3D model eliminates the need for explicit information about the goal position in the reward formulation. Instead, the reward is derived from an anatomical score based on the presence of specific landmarks in the observed image, which were selected based on medical expertise. By minimizing assumptions about the data, this approach not only improves generalization but also enables the use of classical search methods, making the optimization process more feasible.

The navigation system is trained in a simulation environment consisting of 3D labelmap models generated from MRI scans of hands. A labelmap is a 3D representation of specific segmented anatomical structures, specifically those upon which the navigation is based. It is assumed that the navigation can then be implemented on real US images, where the same anatomical structures are visible, through a transfer learning process.

This work presents the development of a modular reinforcement learning environment, which simulates the navigation task using 3D labelmap models generated from MRI scans of hands. These scans represent specific segmented anatomical structures crucial for the proposed anatomical score. Different observation models have been formulated within this environment, offering varying levels of abstraction to explore the impact of different perceptual inputs. The modular design facilitates the rapid testing of these observations. It is assumed that, through transfer learning, the navigation system can be adapted for real ultrasound images where the same anatomical structures are visible and can be segmented.

To summarize, the method proposed in this work operates on 3D labelmaps of the selected tissues. The anatomical features of interest must thus be segmented from a 3D image volume. The agent learns to navigate through the model by taking 2D slices with arbitrary orientation. It uses a clustering method to recognize the clusters of features and their position, which are needed to infer the reward function. Given the reward for each slice position, and assuming the presence of a globally optimal region, it is possible to solve the search as an optimization problem, navigating to the optimal region of the carpal tunnel.

The main contributions of this thesis are:

- A reinforcement learning solution for learning autonomous navigation of 2D images in a 3D model, which does not require any information about the goal position for training.

- An observable reward function based on the presence of anatomical features in the currently observed image, which has been previously selected upon medical feedback.

- An open-source reinforcement learning simulation environment consisting of 3D labeled models generated from MRI scans of hands, with a focus on modularity and maintainability.

Thesis outline
--------------

This thesis is divided into seven chapters. The introduction has provided the medical motivation for the project, introduced the research question, and outlined the objectives of the thesis. The following chapter, :ref:`ch02`, offers essential background information on the medical and technical aspects of the project, including the anatomy of the carpal tunnel, the medical imaging techniques used to assess it, and the robotic systems that assist in medical diagnosis. Additionally, it will review related approaches to autonomous navigation in medical imaging. :ref:`ch03` explains the fundamentals of reinforcement learning, presenting the key concepts and algorithms employed in this work and in the related literature. :ref:`ch04` describes the materials and methods used, including the dataset preparation, the labeling process, and the image processing techniques used to extract image slices from the volume. This chapter also introduces the anatomy-based reward function and the clusteringmethod used to detect relevant anatomical landmarks. :ref:`ch05` details the API of the environment developed for training the reinforcement learning agent, along with the implementation of the agent and the learning algorithms. :ref:`ch06` presents the experimental results that evaluate the agent’s performance, while :ref:`ch07` concludes the thesis by summarizing the main findings and contributions, discussing the work’s limitations, and proposing future research directions.




Bibliography
------------

.. footbibliography::
