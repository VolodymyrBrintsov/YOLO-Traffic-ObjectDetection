import tensorflow as tf

def yolo_model(input_shape, num_classes):
    backbone = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    backbone.trainable = False
    last_layer = backbone.output
    x = tf.keras.layers.Conv2D(2048, 1, activation='relu')(last_layer)
    x = tf.keras.layers.Conv2D(2048, 1, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    objectness_score = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    # Output layers for bounding box coordinates
    bounding_box = tf.keras.layers.Conv2D(4, 1)(x)

    # Output layers for class probabilities
    class_probabilities = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(x)

    # Combine the predictors
    output = tf.keras.layers.Concatenate()([bounding_box, objectness_score, class_probabilities])

    model = tf.keras.Model(inputs=backbone.input, outputs=output)

    return model

def yolo_loss(y_true, y_pred, num_classes, grid, lambda_coord=5.0, lambda_noobj=0.5):
    # Reshape y_pred and y_true to match the grid cell format
    y_pred = tf.cast(tf.reshape(y_pred, (-1, grid[0], grid[1], 5+num_classes)), tf.float32)
    y_true = tf.cast(tf.reshape(y_true, (-1, grid[0], grid[1], 5+num_classes)), tf.float32)

    # Split the predictions and ground truth
    pred_box_coords = y_pred[..., :4]
    pred_obj_confidence = y_pred[..., 4]
    pred_class_probs = y_pred[..., 5:]

    true_box_coords = y_true[..., :4]
    true_obj_confidence = y_true[..., 4]
    true_class_probs = y_true[..., 5:]

    # Calculate the smooth L1 loss for box coordinates
    box_coords_diff = (pred_box_coords - true_box_coords)
    box_coords_loss = tf.reduce_sum(tf.where(tf.abs(box_coords_diff) < 1.0, 0.5 * tf.square(box_coords_diff), tf.abs(box_coords_diff)), axis=-1)
    box_coords_loss = tf.reduce_sum(true_obj_confidence * box_coords_loss, axis=(1, 2))
    # Calculate the confidence loss (Binary Crossentropy) for object presence
    obj_confidence_loss = tf.keras.losses.binary_crossentropy(true_obj_confidence, pred_obj_confidence)
    obj_confidence_loss = tf.reduce_sum(obj_confidence_loss)
    # Calculate the confidence loss (Binary Crossentropy) for no object presence
    noobj_confidence_loss = tf.keras.losses.binary_crossentropy(1 - true_obj_confidence, 1 - pred_obj_confidence)
    noobj_confidence_loss = tf.reduce_sum(noobj_confidence_loss)

    # Calculate the class prediction loss (Binary Crossentropy)
    class_probs_loss = tf.keras.losses.binary_crossentropy(true_class_probs, pred_class_probs)
    class_probs_loss = tf.reduce_sum(true_obj_confidence * class_probs_loss)
    # Calculate the total loss
    total_loss = lambda_coord * tf.reduce_sum(box_coords_loss) + \
                 tf.reduce_sum(true_obj_confidence * obj_confidence_loss) + \
                 lambda_noobj * tf.reduce_sum((1 - true_obj_confidence) * noobj_confidence_loss) + \
                 tf.reduce_sum(class_probs_loss)
    # Compute the mean loss across batches
    mean_loss = total_loss / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    return mean_loss

def iou(box1, box2):
    # Calculate the Intersection over Union (IoU) between two bounding boxes
    x1 = tf.maximum(box1[..., 0], box2[..., 0])
    y1 = tf.maximum(box1[..., 1], box2[..., 1])
    x2 = tf.minimum(box1[..., 2], box2[..., 2])
    y2 = tf.minimum(box1[..., 3], box2[..., 3])

    intersection_area = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    iou = intersection_area / (box1_area + box2_area - intersection_area + 1e-8)
    return iou

def yolo_accuracy(y_true, y_pred, num_classes, grid):
    # Reshape y_pred and y_true to match the grid cell format
    y_pred = tf.cast(tf.reshape(y_pred, (-1, grid[0], grid[1], 5+num_classes)), tf.float32)
    y_true = tf.cast(tf.reshape(y_true, (-1, grid[0], grid[1], 5+num_classes)), tf.float32)

    # Split the predictions and ground truth
    pred_box_coords = y_pred[..., :4]
    pred_obj_confidence = y_pred[..., 4]
    pred_class_probs = y_pred[..., 5:]

    true_box_coords = y_true[..., :4]
    true_obj_confidence = y_true[..., 4]
    true_class_probs = y_true[..., 5:]

    # Calculate binary accuracy for object presence
    obj_accuracy = tf.reduce_mean(tf.keras.metrics.binary_accuracy(true_obj_confidence, pred_obj_confidence))

    # Convert class probabilities to class predictions (argmax)
    pred_class_labels = tf.argmax(pred_class_probs, axis=-1)
    true_class_labels = tf.argmax(true_class_probs, axis=-1)

    # Calculate classification accuracy only for grid cells with objects (true_obj_confidence == 1)
    mask = tf.cast(true_obj_confidence, tf.bool)
    class_accuracy = tf.reduce_sum(tf.cast(tf.equal(pred_class_labels, true_class_labels), tf.float32) * tf.cast(mask, tf.float32))
    class_accuracy /= (tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-8)

    # Calculate bounding box accuracy using Intersection over Union (IoU)
    iou_scores = iou(pred_box_coords, true_box_coords)
    box_accuracy = tf.reduce_sum(true_obj_confidence * tf.cast(iou_scores > 0.5, tf.float32)) / (tf.reduce_sum(true_obj_confidence) + 1e-8)
    # Overall accuracy (weighted average of object accuracy, class accuracy, and box accuracy)
    overall_accuracy = (obj_accuracy + class_accuracy + box_accuracy) / 3.0

    return overall_accuracy

def define_and_compile_model(num_classes, grid):
  model = yolo_model((224, 224, 3), num_classes)

  model.compile(optimizer='adam', loss=lambda y_true, y_pred: yolo_loss(y_true, y_pred, num_classes, grid), metrics=[lambda y_true, y_pred: yolo_accuracy(y_true, y_pred, num_classes, grid)])
  return model