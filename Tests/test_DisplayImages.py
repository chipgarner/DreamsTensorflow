import DisplayImages
import cv2


def test_show_image():
    di = DisplayImages.DisplayImages(False)
    image = cv2.imread('ImagesIn/Wet.JPG')

    di.show_image(image, 1)

    di.close()


def test_show_image_full_screen():
    di = DisplayImages.DisplayImages()
    image = cv2.imread('ImagesIn/Wet.JPG')

    di.show_image(image, 1)

    di.close()


def test_get_screen_resolution():
    di = DisplayImages.DisplayImages()

    assert di.resolution['height'] == 1080
    assert di.resolution['width'] == 1920

    di.close()


def test_get_image_resolution():
    di = DisplayImages.DisplayImages(False)
    image = cv2.imread('ImagesIn/Wet.JPG')

    resolution = di.get_image_resolution(image)

    assert resolution['height'] == 3530
    assert resolution['width'] == 3348

    di.close()


def test_pad_image_to_screen_aspect():
    di = DisplayImages.DisplayImages()
    image = cv2.imread('ImagesIn/Wet.JPG')
    resolution_in = di.get_image_resolution(image)

    image = di.pad_image_to_screen_aspect(image)

    resolution_out = di.get_image_resolution(image)
    di.show_image(image, 1000)

    assert resolution_in['width'] != resolution_out['width']

    di.close()

