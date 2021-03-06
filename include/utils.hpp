#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <cstdint>
#include <cstddef>

/**
 * Obtains images and labels from a UByte dataset. If "data" and "labels" are null,
 * only the number of images is returned.
 * 
 * @param image_filename The dataset file containing the images.
 * @param label_filename The dataset file containing the labels.
 * @param data The output dataset, a Dx1xHxW array (single channel, D is the dataset size).
 * @param labels The Dx1 label array.
 * @param width The width of each image.
 * @param height The height of each image.
 * @return Number of images in dataset.
 */
size_t ReadUByteDataset(const char* image_filename, const char* label_filename, 
                        uint8_t *data, uint8_t *labels, size_t& width, size_t& height);

#endif  // _UTILS_HPP_
