import cv2
import numpy as np
import subprocess


class DisplayImages:
    def __init__(self, full_screen=True):
        self.screen_aspect_ratio = 1.0
        self.resolution = None

        if full_screen:
            cv2.namedWindow("Show", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Show", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            self.resolution = self.__get_screen_resolution()
            self.screen_aspect_ratio = self.resolution['width'] / self.resolution['height']

    def show_image(self, image, wait=1):
        cv2.imshow('Show', image)
        cv2.waitKey(wait)

    def close(self):
        cv2.destroyAllWindows()

    def __get_screen_resolution(self):
        output = \
        subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4', shell=True, stdout=subprocess.PIPE).communicate()[0]
        resolution = output.split()[0].split(b'x')
        return {'width': int(resolution[0]), 'height': int(resolution[1])}

    @staticmethod
    def get_image_resolution(image):
        height, width = image.shape[:2]
        return {'width': width, 'height': height}

    def pad_image_to_screen_aspect(self, image):
        height, width = image.shape[:2]
        aspect_ratio = width / height
        if self.screen_aspect_ratio - aspect_ratio > 0.001: # Only for screen aspect wider than image
            image_width = round(height * self.screen_aspect_ratio)
            edges = np.zeros((height, image_width, 3), np.uint8)
            offset = (image_width - width) // 2
            edges[0:height, offset:image_width - offset] = image
        else:
            edges = image
        return edges




