"""We use shape naming conventions in this module (applied to all variables).

hw: height, width. Sometimes also called y_size and x_size
yx: y, x (positions).
"""


import numpy as np


class IncompatibleShapeError(Exception):
    pass


def crop_center(img_hw: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Perform a center crop of the input image to the specified shape.

    :param img_hw: a 2d array to be cropped
    :param shape_hw: Shape of the cropped image
    """
    if img_hw.shape == tuple(shape_hw):
        return img_hw
    img_bbox = _validate_and_get_full_image_bbox(img_hw, shape_hw)
    crop_y_pos, crop_x_pos, crop_y_size, crop_x_size = centered_constrained_bbox(
        limiting_bbox_yxhw=img_bbox,
        new_bbox_shape_hw=shape_hw,
    )
    return img_hw[crop_y_pos : crop_y_pos + crop_y_size, crop_x_pos : crop_x_pos + crop_x_size]


def _validate_and_get_full_image_bbox(
    img_hw: np.ndarray,
    shape_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Helper function for validating input and returning a bounding box containing the full image, i.e.
    (0, 0, img_w, img_h). Useful for processing images with `centered_constrained_bbox`.
    """
    if np.any(np.array(shape_hw) < 0):
        raise ValueError("shape cannot contain negative entries.")
    y_size, x_size = img_hw.shape
    new_y_size, new_x_size = shape_hw
    if y_size < new_y_size or x_size < new_x_size:
        raise IncompatibleShapeError(
            f"Center crop shape, {shape_hw}, larger than provided image of shape, {img_hw.shape}",
        )
    return 0, 0, y_size, x_size


def centered_constrained_bbox(
    limiting_bbox_yxhw: tuple[int, int, int, int],
    new_bbox_shape_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Returns a bounding box centered on and contained within the limiting bounding box with the specified shape.
    A bounding box is a tuple of type (y_pos, x_pos, height, width) where (y_pos, x_pos) point to the
    upper left corner.

    Can be interpreted as "center cropping" the input bounding box to the specified shape.

    :param limiting_bbox_yxhw: Bounding box that new bbox is contained within and centered on
    :param new_bbox_shape_hw: Shape of centered bounding box
    :return: Centered bounding box
    """
    y_pos, x_pos, y_size, x_size = limiting_bbox_yxhw
    new_y_size, new_x_size = new_bbox_shape_hw

    if np.any(np.array([y_size, x_size, new_y_size, new_x_size]) < 0):
        raise ValueError(
            "Limiting bounding box shape and new bounding box shape cannot contain negative entries.",
        )

    if y_size < new_y_size or x_size < new_x_size:
        raise IncompatibleShapeError(
            f"Requested shape of centered bounding box, {new_bbox_shape_hw}, has to be smaller than provided limiting "
            f"bounding box shape, {limiting_bbox_yxhw[2:]}",
        )

    # + 1 ensures that this uses the same convention for asymmetric cropping as generally used for convolutions
    # that is if we want to center crop [0, 1, 2, 3, 4] down to length 2 we crop it to [2, 3], i.e. the discarded
    # interval is chosen to be larger on the "left"
    new_y_pos = y_pos + (y_size - new_y_size + 1) // 2
    new_x_pos = x_pos + (x_size - new_x_size + 1) // 2

    return new_y_pos, new_x_pos, new_y_size, new_x_size
