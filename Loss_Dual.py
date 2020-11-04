import tensorflow as tf
from Attention_cnn.model.Network import gradient_mag


def loss_bce(y_true, y_pred, eps=0.0, from_logits=True):
    

    # [b, h, w, classes]
    if from_logits:
        y_pred = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(y_pred, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / counts**2
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights * multed, axis=-1)
    denom = tf.reduce_sum(weights * summed, axis=-1)
    bce = 1. - 2. * numerators / denom
    bce = tf.where(tf.math.is_finite(bce), bce, tf.zeros_like(bce))
    return tf.reduce_mean(bce)


def _gumbel_softmax(logits, eps=1e-8, tau=1.):
    """

    :param logits:
    :param eps:
    :param tau temprature:
    :return soft approximation to argmax:

    see https://arxiv.org/abs/1611.01144
    """
    g = tf.random.uniform(tf.shape(logits))
    g = -tf.math.log(eps - tf.math.log(g + eps))
    return tf.nn.softmax((logits + g) / tau)


def _segmentation_edge_loss(gt_tensor, logit_tensor, thresh=0.8):
   
    
    logit_tensor = _gumbel_softmax(logit_tensor)

   
    gt_edges = gradient_mag(gt_tensor)
    pred_edges = gradient_mag(logit_tensor)

    # [b*h*w, n]
    gt_edges = tf.reshape(gt_edges, [-1, tf.shape(gt_edges)[-1]])
    pred_edges = tf.reshape(pred_edges, [-1, tf.shape(gt_edges)[-1]])

   
    edge_difference = tf.abs(gt_edges - pred_edges)

    # gt edges and disagreement with pred
    mask_gt = tf.cast((gt_edges > thresh ** 2), tf.float32)
    contrib_0 = tf.boolean_mask(edge_difference, mask_gt)

    contrib_0 = tf.cond(
        tf.greater(tf.size(contrib_0), 0),
        lambda: tf.reduce_mean(contrib_0),
        lambda: 0.)

   
    mask_pred = tf.stop_gradient(tf.cast((pred_edges > thresh ** 2), tf.float32))
    contrib_1 = tf.reduce_mean(tf.boolean_mask(edge_difference, mask_pred))
    contrib_1 = tf.cond(
        tf.greater(tf.size(contrib_1), 0),
        lambda: tf.reduce_mean(contrib_1),
        lambda: 0.)
    return tf.reduce_mean(0.5 * contrib_0 + 0.5 * contrib_1)


def _shape_edge_loss(gt_tensor, pred_tensor, pred_shape_tensor, keep_mask, thresh=0.8):
    
    mask = pred_shape_tensor > thresh
    mask = tf.stop_gradient(mask[..., 0])
    mask = tf.logical_and(mask, keep_mask)

    # get relavent predicitons and truth
    gt = gt_tensor[mask]
    pred = pred_tensor[mask]

    
    if tf.reduce_sum(tf.cast(mask, tf.float32)) > 0:
        return tf.reduce_mean(tf.losses.binary_crossentropy(gt, pred, from_logits=True))
    else:
        return 0.


def _weighted_cross_entropy(y_true, y_pred, keep_mask):
    
    y_true = y_true[keep_mask]
    y_pred = y_pred[keep_mask]

    # weights
    rs = tf.reduce_sum(y_true, axis=0, keepdims=True)
    N = tf.cast(tf.shape(y_true)[0], tf.float32)
    weights = (N - rs)/N + 1

    
    weights = tf.reduce_sum(y_true*weights, axis=1)

 
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    weighted_losses = unweighted_losses * weights
    loss = tf.reduce_mean(weighted_losses)
    return loss


def loss(gt_label, logits, shape_head, edge_label, loss_weights):
    tf.debugging.assert_shapes([
        (gt_label,     ('b', 'h', 'w', 'c')),
        (logits,       ('b', 'h', 'w', 'c')),
        (shape_head,   ('b', 'h', 'w', 1)),
        (edge_label,   ('b', 'h', 'w', 2)),
        (loss_weights, (4,))],)

   
    keep_mask = tf.reduce_any(gt_label == 1., axis=-1)
    anything_active = tf.reduce_any(keep_mask)

    
    seg_loss = tf.cond(
        anything_active,
        lambda: _weighted_cross_entropy(gt_label, logits, keep_mask) * loss_weights[0],
        lambda: 0.)

    
    shape_probs = tf.concat([1. - shape_head, shape_head], axis=-1)
    edge_loss = loss_bce(edge_label, shape_probs) * loss_weights[1]

    edge_consistency = _segmentation_edge_loss(gt_label, logits) * loss_weights[2]
    # this ensures that the classifcatiomn at the edges is correct
    edge_class_consistency = tf.cond(
        anything_active,
        lambda: _shape_edge_loss(gt_label, logits, shape_head, keep_mask) * loss_weights[3],
        lambda: 0.)
    return seg_loss, edge_loss, edge_class_consistency, edge_consistency

