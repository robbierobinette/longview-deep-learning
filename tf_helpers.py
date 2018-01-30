import tensorflow as tf


def log_variable(var: tf.Variable):
    with tf.name_scope('variable-summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_dense_norm(input: tf.Tensor, n_out: int, layer_norm: bool, keep_prob: tf.Tensor = None,
                   name="dense_with_layer_norm"):
    with tf.name_scope(name):
        n_in = int(input.shape[1])
        w = tf.get_variable(name="%s-W" % name, shape=[n_in, n_out], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="%s-b" % name, shape=[n_out], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        # log_variable(w)
        # log_variable(b)

        y = tf.matmul(input, w) + b
        if (layer_norm):
            y = tf.contrib.layers.layer_norm(inputs=y, center=True, scale=True)
        y = tf.nn.relu(y, name="relu")
        if (keep_prob != None):
            y = tf.nn.dropout(y, keep_prob)
        return y


def add_residual_block(input: tf.Tensor, block_width: int, n_layers: int, layer_norm: bool, keep_prob: tf.Tensor = None,
                       name="residual_block"):
    input_width = int(input.shape[1])

    # add N - 1 layers of shape n_in x block_width
    tip = input
    for i in range(n_layers - 1):
        tip = add_dense_norm(tip, block_width, layer_norm, keep_prob, name=("%s-%d" % (name, i)))

    # add a final layer converting back to the original input_width
    w = tf.get_variable(name="%s-w-final" % name, shape=[block_width, input_width], dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
    # log_variable(w)
    y = tf.matmul(tip, w)
    y1 = y + input
    y2 = tf.nn.relu(y1, name="y2")
    return y2


def add_dense(input: tf.Tensor, n_out: int, name="dense"):
    with tf.name_scope(name):
        n_in = int(input.shape[1])
        w = tf.get_variable(name="%s-W" % name, shape=[n_in, n_out], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="%s-b" % name, shape=[n_out], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        # log_variable(w)
        # log_variable(b)
        y = tf.matmul(input, w) + b
        y2 = tf.nn.relu(y, name="relu")
        return y2
