import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy

def weighted_crossentropy(weights):
    return lambda y_true, y_pred: categorical_crossentropy(y_true * weights, y_pred)

def dice(y_true, y_pred):
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
    dice = 2 * intersection / (union + 1)
    return dice

def dice_loss(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    dice = 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)
    return dice

def probability_dice_of_a_class(y_true, y_pred, class_index):
    y_true = K.flatten(y_true[..., class_index])
    y_pred = K.flatten(y_pred[..., class_index])

    y_true = tf.to_float(tf.logical_not(tf.equal(y_true, 1)))

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice = (2.0 * intersection + K.epsilon()) / (union + K.epsilon())
    return dice

def dice_of_a_class(y_true, y_pred, class_index):
    y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    y_true = tf.to_int32(tf.equal(y_true, class_index))
    y_pred = tf.to_int32(tf.equal(y_pred, class_index))

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(y_pred, y_true), tf.int32), y_true)), tf.float32)
    union = tf.count_nonzero(y_pred, dtype=tf.float32) + tf.count_nonzero(y_true, dtype=tf.float32)
    # dice = tf.cond(tf.logical_and(tf.equal(K.sum(truelabels_bool), 0), tf.equal(K.sum(predictions_bool), 0)), lambda: 1.0, lambda: 2 * intersection / (union + 1))
    dice = (2.0 * intersection + K.epsilon()) / (union + K.epsilon())
    return dice

def probability_precision_of_a_class(y_true, y_pred, class_index):
    y_true = K.flatten(y_true[..., class_index])
    y_pred = K.flatten(y_pred[..., class_index])

    y_true = tf.to_float(tf.logical_not(tf.equal(y_true, 1)))

    intersection = K.sum(y_true * y_pred)
    dice = intersection / K.sum(y_true)
    return dice

def recall_of_a_class(y_true, y_pred, class_index):
    y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    y_true = tf.to_int32(tf.equal(y_true, class_index))
    y_pred = tf.to_int32(tf.equal(y_pred, class_index))

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(y_pred, y_true), tf.int32), y_true)), tf.float32)
    dice = (intersection + K.epsilon()) / (tf.count_nonzero(y_true, dtype=tf.float32) + K.epsilon())
    return dice

def probaility_precision_of_a_class(y_true, y_pred, class_index):
    y_true = K.flatten(y_true[..., class_index])
    y_pred = K.flatten(y_pred[..., class_index])

    y_true = tf.to_float(tf.logical_not(tf.equal(y_true, 1)))

    intersection = K.sum(y_true * y_pred)
    dice = intersection / K.sum(y_pred)
    return dice

def precision_of_a_class(y_true, y_pred, class_index):
    y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    y_true = tf.to_int32(tf.equal(y_true, class_index))
    y_pred = tf.to_int32(tf.equal(y_pred, class_index))

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(y_pred, y_true), tf.int32), y_true)), tf.float32)
    dice = (intersection + K.epsilon()) / (tf.count_nonzero(y_pred, dtype=tf.float32) + K.epsilon())
    return dice

def bg_probability_dice(y_true, y_pred):
    return probability_dice_of_a_class(y_true, y_pred, 0)

def bg_dice(y_true, y_pred):
    return dice_of_a_class(y_true, y_pred, 0)

def bg_probability_recall(y_true, y_pred):
    return probability_precision_of_a_class(y_true, y_pred, 0)

def bg_recall(y_true, y_pred):
    return recall_of_a_class(y_true, y_pred, 0)

def bg_probability_precision(y_true, y_pred):
    return probability_precision_of_a_class(y_true, y_pred, 0)

def bg_precision(y_true, y_pred):
    return precision_of_a_class(y_true, y_pred, 0)

def hcc_probability_dice(y_true, y_pred):
    return probability_dice_of_a_class(y_true, y_pred, 1)

def hcc_dice(y_true, y_pred):
    return dice_of_a_class(y_true, y_pred, 1)

def hcc_probability_recall(y_true, y_pred):
    return probability_precision_of_a_class(y_true, y_pred, 1)

def hcc_recall(y_true, y_pred):
    return recall_of_a_class(y_true, y_pred, 1)

def hcc_probability_precision(y_true, y_pred):
    return probability_precision_of_a_class(y_true, y_pred, 1)

def hcc_precision(y_true, y_pred):
    return precision_of_a_class(y_true, y_pred, 1)

def cyst_probability_dice(y_true, y_pred):
    return probability_dice_of_a_class(y_true, y_pred, 2)

def cyst_dice(y_true, y_pred):
    return dice_of_a_class(y_true, y_pred, 2)

def cyst_probability_recall(y_true, y_pred):
    return probability_precision_of_a_class(y_true, y_pred, 2)

def cyst_recall(y_true, y_pred):
    return recall_of_a_class(y_true, y_pred, 2)

def cyst_probability_precision(y_true, y_pred):
    return probability_precision_of_a_class(y_true, y_pred, 2)

def cyst_precision(y_true, y_pred):
    return precision_of_a_class(y_true, y_pred, 2)

def angioma_probability_dice(y_true, y_pred):
    return probability_dice_of_a_class(y_true, y_pred, 3)

def angioma_dice(y_true, y_pred):
    return dice_of_a_class(y_true, y_pred, 3)

def angioma_probability_recall(y_true, y_pred):
    return probability_precision_of_a_class(y_true, y_pred, 3)

def angioma_recall(y_true, y_pred):
    return recall_of_a_class(y_true, y_pred, 3)

def angioma_probability_precision(y_true, y_pred):
    return probability_precision_of_a_class(y_true, y_pred, 3)

def angioma_precision(y_true, y_pred):
    return precision_of_a_class(y_true, y_pred, 3)
