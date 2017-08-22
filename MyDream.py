# boilerplate code
import numpy as np
import random
import cv2
import tensorflow as tf

model_fn = 'inception5h/tensorflow_inception_graph.pb'

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')  # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]

print('Number of layers', len(layers))
print('Total number of feature channels:', sum(feature_nums))


# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139  # picking some feature channel to visualize

# start with a gray image with a little noise
img_noise = np.zeros(shape=(224, 224, 3)) + 100.0  # .random.uniform(size=(224, 224, 3)) + 100.0

cv2.namedWindow("Show", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Show", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def showarray(a):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    cv2.imshow('Show', a)
    cv2.waitKey(1)


def T(layer):
    # Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0" % layer)


def tffunc(*argtypes):
    # Helper that transforms TF-graph generating function into a regular one.
    # See "resize" function below.

    placeholders = list(map(tf.placeholder, argtypes))

    def wrap(f):
        out = f(*placeholders)

        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

        return wrapper

    return wrap


# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0, :, :, :]


resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, tile_size=200):
    # '''Compute the value of tensor t_grad over the image in a tiled way.
    # Random shifts are applied to the image to blur tile boundaries over
    # multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_deepdream(t_obj, img0=img_noise,
                     iter_n=100, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))
            print('.', end=' ')
            showarray(img / 255.0)

    return img


def get_image(path):
    img0 = cv2.imread(path)
    img0 = np.float32(img0)
    return img0


# img0 = get_image('ImagesIn/profile800.jpg')
# render_deepdream(T(layer)[:,:,:,139], img0)
# render_deepdream(tf.square(T('mixed4c')), img0)

img_profile = get_image('ImagesIn/profile740.jpg')
img_parasol = get_image('ImagesIn/parasolSmall.jpg')
img_eye = get_image('ImagesIn/eye740.jpg')
img_leaves = get_image('ImagesIn/leaves.jpg')
img_roses = get_image('ImagesIn/roses.jpg')

channels = [1, 16, 18, 7, 24, 4, 11, 19, 23, 111, 30, 36, 31, 42, 41, 44, 123, 108, 109, 47, 45, 51, 52, 127, 128, 134,
            57, 58, 53, 60, 139, 140, 112,
            61, 75, 101, 100, 98, 90, 136, 114, 122, 115, 82, 70, 138, 97, 116, 117, 87, 141, 86, 83, 143]

while True:
    for channel in channels:
        which_image = random.randint(1, 5)

        if which_image == 1:
            render_deepdream(T(layer)[:, :, :, channel], img_eye)
        elif which_image == 2:
            render_deepdream(T(layer)[:, :, :, channel], img_parasol)
        elif which_image == 3:
            render_deepdream(T(layer)[:, :, :, channel], img_leaves)
        elif which_image == 4:
            render_deepdream(T(layer)[:, :, :, channel], img_roses)
        else:
            render_deepdream(T(layer)[:, :, :, channel], img_profile)

        # path = 'ImagesOut/mixed' + str(i) + '.jpg'
        # print(path)
        # cv2.imwrite(path, img)
