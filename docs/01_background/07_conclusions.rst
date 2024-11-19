.. _`ch07`:

Conclusions
===========

Discussion
----------

This work addressed the research question of navigating medical images using only anatomical feedback through a Deep Reinforcement Learning (DRL) approach. The main difference with other state-of-the-art approaches is the use of an observable reward function. Current RL methods use a reward function that is not directly observable, which makes it difficult to understand the behavior of the agent when it generalizes to unseen data and might result in some unpredictable behavior. This work proposes a navigation reward based on the anatomical landmarks visible in the currently visualized image. This ensures that the reward is a directly observable objective, improving predictability and generalization to unseen data.

The main contribution of the thesis has been the development of a modular simulation environment, which allows for quick and easy testing of different environments and agent configurations. The environment, built on the Gymnasyum API, provides an interface to train navigation through a 3D model of the human hand. The environment can be instantiated with multiple observations that describe the current state of the environment at different levels of abstraction. Moreover, the framework allows to combine multiple observation spaces with the use of wrappers. The project and detailed documentation of the API is available open source on GitHub [1]_.

Additional contributions of this work include the generation of a labeled image dataset that highlights the key anatomical features used by doctors for navigation to the carpal tunnel. An arbitrary reslicing function has been developed for the 3D model, enabling visualization of image planes in any orientation. Additionally, a clustering algorithm has been implemented to identify the anatomical landmarks in the images.

The experiments conducted in this thesis validate the approach of using an observable reward based on anatomical features to achieve autonomous navigation to the carpal tunnel within a 3D volume. By relying solely on the anatomical landmarks visible in the currently observed image, the proposed method ensures that the reward function remains directly tied to the real-time visual feedback, enhancing both the predictability of the agent’s behavior and its generalization to unseen data.

The simplified experiments using the *ActionRewardObservation* demonstrated that the agent could learn a navigation strategy effectively in a reduced problem space. In the 1D projection experiments, its outcome indicates that the reinforcement learning architecture employed is capable of handling the task provided the exploration space is appropriately constrained.

However, when the complexity of the task was increased by expanding the action space to two dimensions, the agent’s performance declined. While the agent was still able to outperform random search, the learning process was less robust, and the agent often failed to converge to an optimal policy. This suggests that while the reward function based on anatomical features provides valuable guidance, the observation space may not be informative enough in higher-dimensional settings, or the exploration time was insufficient.

Moreover, the experiments highlighted the potential of using a memory stack of the best action-reward pairs to enhance the agent’s decision-making process. This approach showed promise in improving the agent’s performance in 2D navigation, but it also underscored the need for more sophisticated methods to manage the additional complexity introduced by such memory mechanisms.

Limitations
-----------

Throughout the development and experimentation phase of the project, several limitations were identified, that impacted the overall effectiveness and efficiency of the proposed approach. These limitations can be categorized into computational and methodological challenges, that need to be addressed in future work to improve the robustness and generalizability of the proposed method.

Computational Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~

Firstly, the partial observability of the environment posed a significant challenges, particularly in the 2D navigation tasks, where the observation space may not have provided sufficient information for robust policy learning. The reliance on a memory stack to compensate for this limitation was only partially successful, indicating the need for more advanced observation strategies.

Another limitation was the significant computational resources required for training. The high-dimensional exploration space of the 3D volume presented a demanding challenge, requiring the agent to explore a vast range of potential actions’ transitions to find a good policy. Additionally, the Soft Actor-Critic (SAC) algorithm, while well-suited for continuous action spaces, is inherently computationally expensive. This is due to its use of multiple neural networks, including a value-network, two Q-networks for double Q-learning, and a policy network, all of which require frequent gradient updates. Each gradient step involves backpropagation through these networks, which increases the computational load. Although parallelization was employed to distribute the workload across multiple environments, the combination of the large exploration space and the algorithm’s complexity still resulted in prolonged training times and constrained the number of experiments that could be conducted.

Furthermore, the implementation of the multimodal neural network faced challenges related to memory management. The large buffer size required by the Soft Actor-Critic algorithm exceeded the available memory, causing some experiments to fail. This highlights the need for more efficient memory management strategies, particularly when handling high-dimensional data and large replay buffers.

Methodological Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the primary limitations encountered was related to the clustering algorithm used to identify the relevant anatomical features in the MRI-segmented data. The DBSCAN algorithm was employed, which relies on a density function determined by the distance parameter :math:`\epsilon` and the minimum number of neighboring points required to form a cluster. However, the distance :math:`\epsilon` was calculated based on pixel indexes, which poses a challenge for generalization across different volumes with varying resolutions. To improve generalizability, the distance metric should be adapted to reflect actual metric distances by transforming the pixel distances according to the pixel spacing provided by the MRI data. Additionally, the process of finding the optimal parameters for DBSCAN was time-consuming and lacked automation, which would be beneficial for improving efficiency.

An additional challenge for clustering was the limited resolution of the MRI-segmented data, which made it difficult to fully segment individual tendon clusters. Instead, groups of tendons were clustered together, potentially leading to inconsistencies in the clustering results across different volumes. Addressing this issue could involve clustering the entire tendons region and then reasoning about its overall shape and position, rather than relying on individual clusters. While this approach would require specific tuning to ensure accurate representation, it could enhance the robustness of the clustering in scenarios where individual tendon segmentation is not feasible.

In addition to the challenges posed by the clustering algorithm, limitations were also identified in the current reward formulation. Although the reward function effectively identifies images that meet the criteria based on the number and relative positions of clustered features, it may result in selecting slices that, while scoring high, are quite different from those at the carpal tunnel standard plane. This issue arises because the reward function does not fully account for the specific anatomical context that characterizes the carpal tunnel.

To address this, a more sophisticated reward formulation could be developed by incorporating the expected size and shape of the clusters. For instance, including the distinct shape of the hamate bone’s hook, a highly recognizable anatomical feature would help ensure that the selected slices more accurately represent the target region. Furthermore, enhancing the reward function to better assess the relative positions of each cluster would allow for a closer match to the actual anatomical layout of the carpal tunnel. This improvement would enable the agent to distinguish between slices that merely meet the general clustering criteria and those that genuinely represent the desired anatomical region.

Future Work
-----------

Building upon the findings and limitations identified in this work, several key areas for future research can be outlined. These areas aim to address the challenges encountered and further enhance the proposed approach to enable more effective and efficient navigation in medical images. Moreover, future steps should bring the proposed method closer to practical applications to enable real-world deployment in clinical settings.

In regard to the proposed method, the first step would be to refine the clustering process, particularly by adapting the distance metric and the difference in resolution across different volumes. Additionally, automating the parameter selection for the DBSCAN algorithm would improve the efficiency of and enhance the robustness of the clustering process.

The reward function could be further developed to incorporate more detailed anatomical information, such as the expected size and shape of the clusters and the relative positions of the features. This would help ensure that the agent selects slices that more accurately represent the carpal tunnel region, rather than merely meeting the general clustering criteria.

In terms of the algorithmic approach, the experiments demonstrated the effectiveness of the Soft Actor-Critic algorithm in learning navigation policies. However, further research is needed to optimize the algorithm’s parameters. This includes experimenting with different learning rate schedules, as well as different batch sizes and replay buffer sizes, to improve the training efficiency and stability of the algorithm. Moreover, future experiments should be conducted with different random seeds to assess the robustness of the learned policies. This would help determine whether the agent’s performance is consistent across different initializations and highlight potential areas for improvement.

Further work has to be invested into optimizing the environment steps to make the training more efficient, enabling more extensive experimentation. One promising direction is the refinement of the multimodal neural network to better handle complex observations, particularly by making it compatible with the best actions memory stack and improving its memory efficiency. This would enable the agent to learn from image observations of the clusters, and experiment other image-based observations such as raw MRI or labelmap slices, along with action and reward history, to train an agent capable of more complex decision-making processes.

The observability of the reward function presents opportunities for exploring alternative optimization approaches. While this research included an exhaustive search in one dimension, it would be valuable to extend this approach to two dimensions, particularly to validate the effectiveness of the reward function in more complex environments. Additionally, Bayesian optimization could be a promising technique for efficiently navigating the action space due to its ability to balance exploration and exploitation, offering a good benchmark for reinforcement learning methods.

Finally, while the work has made significant strides toward autonomous navigation in medical imaging, there remains a considerable distance to achieving this goal in real-world ultrasound imaging. Future research should focus on transfer learning strategies that utilize the trained models on the same labelmaps used in this study, extending the experiments to include real ultrasound data. This transfer learning approach would be a critical step towards validating the method in practical applications and ensuring its robustness and effectiveness in clinical settings.

.. [1]
   https://github.com/appliedAI-Initiative/armscan_env




Bibliography
------------

.. footbibliography:: 
