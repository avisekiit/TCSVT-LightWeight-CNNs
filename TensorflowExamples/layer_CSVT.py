import tensorflow as tf

def bottleneck(x, filter_shape):
    filters = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    return tf.nn.conv2d(x, filters, [1, 1, 1, 1], padding='SAME')

def branchout(x, filter_shape1, filter_shape2):
    filters1 = tf.get_variable(
        name='weight1',
        shape=filter_shape1,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters2_1 = tf.get_variable(
        name='weight2_1',
        shape=[3, 3, filter_shape2[2], 1],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters2_2 = tf.get_variable(
        name='weight2_2',
        shape=[1, 1, filter_shape2[2], filter_shape2[3]],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)

    w = tf.nn.conv2d(x, filters1, [1, 1, 1, 1], padding='SAME')
    y = tf.nn.separable_conv2d(x, filters2_1, filters2_2, [1, 1, 1, 1], padding='SAME')
    z = tf.concat([w,y],3)
    z = tf.reshape(z,(x.shape[0],x.shape[1],x.shape[2],2*filter_shape1[3]))
    return z

def group_conv_normal(x, filter_shape, groups, stride=1):
    # Currently groups is hardcoded to be  = 8 as per our paper
    # One can change according to their experiment's need.
    # That's why we have flters1_, filters2_,... filters8_.
    # One can write this in a loop to have a clearner code.
    filters1_ = tf.get_variable(
        name='weight1_',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters2_ = tf.get_variable(
        name='weight2_',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters3_ = tf.get_variable(
        name='weight3_',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters4_ = tf.get_variable(
        name='weight4_',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters5_ = tf.get_variable(
        name='weight5_',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters6_ = tf.get_variable(
        name='weight6_',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters7_ = tf.get_variable(
        name='weight7_',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters8_ = tf.get_variable(
        name='weight8_',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    
    ig = filter_shape[2] // groups
    offset = 0
    glist = []
    # We have g1,...g8 due to hard coding of number of groups = 8
    g1 = x[:, :, :, offset:offset+ig] 
    g2 = x[:, :, :, offset+ig:offset+(2*ig)]
    g3 = x[:, :, :, offset+(2*ig):offset+(3*ig)]
    g4 = x[:, :, :, offset+(3*ig):offset+(4*ig)]
    g5 = x[:, :, :, offset+(4*ig):offset+(5*ig)] 
    g6 = x[:, :, :, offset+(5*ig):offset+(6*ig)]
    g7 = x[:, :, :, offset+(6*ig):offset+(7*ig)]
    g8 = x[:, :, :, offset+(7*ig):offset+(8*ig)]
    
    # We have y1,...y8 due to hard coding of number of groups = 8
    y1 = tf.nn.conv2d(g1, filters1_, [1, stride, stride, 1], padding='SAME')
    y2 = tf.nn.conv2d(g2, filters2_, [1, stride, stride, 1], padding='SAME')
    y3 = tf.nn.conv2d(g3, filters3_, [1, stride, stride, 1], padding='SAME')
    y4 = tf.nn.conv2d(g4, filters4_, [1, stride, stride, 1], padding='SAME')
    y5 = tf.nn.conv2d(g5, filters5_, [1, stride, stride, 1], padding='SAME')
    y6 = tf.nn.conv2d(g6, filters6_, [1, stride, stride, 1], padding='SAME')
    y7 = tf.nn.conv2d(g7, filters7_, [1, stride, stride, 1], padding='SAME')
    y8 = tf.nn.conv2d(g8, filters8_, [1, stride, stride, 1], padding='SAME')
     
    z = tf.concat([y1,y2,y3,y4,y5,y6,y7,y8],3)
    z = tf.reshape(z,(x.shape[0],x.shape[1] ,x.shape[2], x.shape[3]))

    return z


def group_conv_dilated(x, filter_shape, groups, dilation):
    # Currently groups is hardcoded to be  = 8 as per our paper
    # One can change according to their experiment's need.
    # That's why we have flters1_, filters2_,... filters8_.
    # One can write this in a loop to have a clearner code.
    filters1 = tf.get_variable(
        name='weight1',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters2 = tf.get_variable(
        name='weight2',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters3 = tf.get_variable(
        name='weight3',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters4 = tf.get_variable(
        name='weight4',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters5 = tf.get_variable(
        name='weight5',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters6 = tf.get_variable(
        name='weight6',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters7 = tf.get_variable(
        name='weight7',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    filters8 = tf.get_variable(
        name='weight8',
        shape=[filter_shape[0], filter_shape[1], filter_shape[2]//groups, filter_shape[3]//groups],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
        
    ig = filter_shape[2] // groups
    offset = 0
    glist = []
    # Due to hard coding of number of groups = 8 we need g1,g2,...g8.
    g1 = x[:, :, :, offset:offset+ig] 
    g2 = x[:, :, :, offset+ig:offset+(2*ig)]
    g3 = x[:, :, :, offset+(2*ig):offset+(3*ig)]
    g4 = x[:, :, :, offset+(3*ig):offset+(4*ig)]
    g5 = x[:, :, :, offset+(4*ig):offset+(5*ig)] 
    g6 = x[:, :, :, offset+(5*ig):offset+(6*ig)]
    g7 = x[:, :, :, offset+(6*ig):offset+(7*ig)]
    g8 = x[:, :, :, offset+(7*ig):offset+(8*ig)]

    y1 = tf.nn.atrous_conv2d(g1, filters1, dilation, padding='SAME')
    y2 = tf.nn.atrous_conv2d(g2, filters2, dilation, padding='SAME')
    y3 = tf.nn.atrous_conv2d(g3, filters3, dilation, padding='SAME')
    y4 = tf.nn.atrous_conv2d(g4, filters4, dilation, padding='SAME')
    y5 = tf.nn.atrous_conv2d(g5, filters5, dilation, padding='SAME')
    y6 = tf.nn.atrous_conv2d(g6, filters6, dilation, padding='SAME')
    y7 = tf.nn.atrous_conv2d(g7, filters7, dilation, padding='SAME')
    y8 = tf.nn.atrous_conv2d(g8, filters8, dilation, padding='SAME')
     
    z = tf.concat([y1,y2,y3,y4,y5,y6,y7,y8],3)
    z = tf.reshape(z,(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))

    return z


def channel_shuffle(x, groups):
    y = tf.reshape(x, (x.shape[0], x.shape[1], x.shape[2], groups, x.shape[3]//groups))
    y = tf.transpose(y, perm=[0, 1, 2, 4, 3])
    y = tf.reshape(y, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
    
    return y


def batch_normalize(x, is_training, decay=0.99, epsilon=0.001):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=True)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=True)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)

    return tf.cond(is_training, bn_train, bn_inference)
