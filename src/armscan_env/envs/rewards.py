# my custom loss function for image navigation
from typing import Any

import numpy as np


def anatomy_based_rwd(tissue_clusters: dict, n_landmarks: list = [5, 2, 1]) -> np.ndarray[Any, np.dtype[
    np.floating[Any]]]:
    """Calculate the reward based on the presence and location of anatomical landmarks
    :param tissue_clusters: dictionary of tissues and their clusters
    :param n_landmarks: number of landmarks for each tissue
    :return: reward value.
    """
    print("####################################################")
    print("Calculating loss function:")

    # Presence of landmark tissues:

    bones_loss = abs(len(tissue_clusters["bones"]) - n_landmarks[0])
    ligament_loss = abs(len(tissue_clusters["tendons"]) - n_landmarks[1])
    ulnar_loss = abs(len(tissue_clusters["ulnar"]) - n_landmarks[2])

    landmark_loss = bones_loss + ligament_loss + ulnar_loss

    # Absence of landmarks:
    missing_landmark_loss = 0

    # Location of landmarks:
    location_loss = 1

    # There must be bones:
    if len(tissue_clusters["bones"]) != 0:
        # Get centers of tissue clusters:
        bones_centers = [cluster["center"] for _, cluster in enumerate(tissue_clusters["bones"])]
        bones_centers_mean = np.mean(bones_centers, axis=0)

        # There must be tendons:
        if len(tissue_clusters["tendons"]) != 0:
            # Get centers of tissue clusters:
            ligament_centers = [
                cluster["center"] for _, cluster in enumerate(tissue_clusters["tendons"])
            ]
            ligament_centers_mean = np.mean(ligament_centers, axis=0)

            # Check the orientation of the arm:
            # The bones center might be over or under the tendons center depending on the origin
            if bones_centers_mean[0] > ligament_centers_mean[0]:
                print("Orientation: bones over tendons")
                orientation = -1
            else:
                print("Orientation: bones under tendons")
                orientation = 1

            # There must be one ulnar artery:
            if len(tissue_clusters["ulnar"]) == 1:
                # There must be only one ulnar tissue so there is no need to take the mean
                ulnar_center = tissue_clusters["ulnar"][0]["center"]

                # Ulnar artery must be over tendons in the positive orientation:
                if orientation * ulnar_center[0] > orientation * ligament_centers_mean[0]:
                    location_loss = 0
                else:
                    print("Ulnar center not where expected")

            # if no ulnar artery
            else:
                missing_landmark_loss = 1
                print("No ulnar artery found")
        # if no tendons
        else:
            missing_landmark_loss = 2
            print("No tendons found")
    # if no bones:
    else:
        missing_landmark_loss = 3
        print("No bones found")

    # Loss is bounded between 0 and 1
    loss = (1 / 3) * (0.1 * landmark_loss + (1 / 3) * missing_landmark_loss + location_loss)

    print(f"Landmark loss: {landmark_loss}")
    print(f"Missing landmark loss: {missing_landmark_loss}")
    print(f"Location loss: {location_loss}")
    print(f"Total loss: {loss}")

    print("#################################################### \n")

    return loss
