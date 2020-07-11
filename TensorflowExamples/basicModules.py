from layer_CSVT import branchout, bottlenect, group_conv_dilated, group_conv_normal, batch_normalize

# In this file we are releasing the barebone skeleton of our modules.
# Users can takes these code snippets and plug in into their own
# realizations of LIST, GSAT, up/down sampling layers.

#============================================================
#*** LIST Module ***
# in_channel : number of channels taken as input to the the LIST layer
# out_channel: number of channels that will be output from the LIST layer
    with tf.variable_scope('conv1_1'):
        x = bottleneck(x,[1, 1, in_channel, in_channel//4]) 
        x = batch_normalize(x, is_training)
        x = tf.nn.relu(x)
    with tf.variable_scope('conv1_2'):
        x = branchout(x,[1, 1, in_channel//4, out_channel//2],[3, 3, in_channel//4, out_channel//2]) 
        x = batch_normalize(x, is_training)
        x = tf.nn.relu(x)
#===========================================================


#===========================================================

#*** GSAT Module ***
# in_channel : number of channels taken as input to the the GSAT layer
# num_groups = Number of groups for Group Connvolution (see GSAT section in paper)
    with tf.variable_scope('gsat'):
        skip = x;
        x = group_conv_dilated(x,[3, 3, in_channel, in_channel], num_groups, dilation_factor)
        x = channel_shuffle(x, num_groups)
        x = group_conv_normal(x,[1, 1, in_channel, in_channel], num_groups)
        x = batch_normalize(x,is_training)
        x = skip + x
        x = tf.nn.relu(x)
#===========================================================


#===========================================================
#*** UPSAMPLING Module ***
#/////////////////////////////
# First deterministic upsampling then follow with a LIST layer.
    with tf.variable_scope('deconv1_1'):
        x = tf.image.resize_bilinear(x, [x.shape[1]*stride, x.shape[2]*stride])
        x = bottleneck(x,[1, 1, in_channel, in_channel//4]) 
        x = batch_normalize(x, is_training)
        x = tf.nn.relu(x)
    with tf.variable_scope('deconv1_2'):
        x = branchout(x,[1, 1, in_channel//4, out_channel//2],[3, 3, in_channel//4, out_channel//2]) 
        x = batch_normalize(x, is_training)
        x = tf.nn.relu(x)
#===========================================================


#===========================================================
#*** DOWNSAMPLING Module ***
#////////////////////////////////
# First deterministic downsampling then follow with a LIST layer.
    with tf.variable_scope('conv1_1'):
        x = tf.image.resize_bilinear(x, [x.shape[1]//stride, x.shape[2]//stride])
        x = bottleneck(x,[1, 1, in_channel, in_channel//4]) 
        x = batch_normalize(x, is_training)
        x = tf.nn.relu(x)
    with tf.variable_scope('conv1_2'):
        x = branchout(x,[1, 1, in_channel//4, out_channel//2], [3, 3, in_channel//4, out_channel//2]) 
        x = batch_normalize(x, is_training)
        x = tf.nn.relu(x)
#===========================================================
