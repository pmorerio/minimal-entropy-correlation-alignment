import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np


class logDcoral(object):

    def __init__(self, mode='training', method='baseline', hidden_size = 128, learning_rate=0.0001, batch_size=256, alpha=1.0):
        self.mode=mode
        self.method=method
	self.learning_rate = learning_rate
	self.hidden_repr_size = hidden_size
	self.batch_size = batch_size
	self.alpha = alpha
    
		    
    def E(self, images, is_training = False, reuse=False):
	
	if images.get_shape()[3] == 3:
	    images = tf.image.rgb_to_grayscale(images)
	
	with tf.variable_scope('encoder',reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
		with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
		    net = slim.conv2d(images, 64, 5, scope='conv1')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
		    net = slim.conv2d(net, 128, 5, scope='conv2')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
		    net = tf.contrib.layers.flatten(net)
		    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
		    net = slim.dropout(net, 0.5, is_training=is_training)
		    net = slim.fully_connected(net, self.hidden_repr_size, activation_fn=tf.tanh,scope='fc4')
		    # dropout here or not?
		    #~ net = slim.dropout(net, 0.5, is_training=is_training)
		    return net
    
    def logits(self, inputs, is_training = False, reuse=False):
	
	with tf.variable_scope('logits',reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], activation_fn=None):
		
		return slim.fully_connected(inputs, 10, activation_fn=None, scope='fc5')
	
    
    def coral_loss(self, h_src, h_trg, gamma=1e-3):
	# regularized covariances (D-Coral is not regularized actually..)
	# First: subtract the mean from the data matrix
	batch_size = tf.to_float(tf.shape(h_src)[0])
	h_src = h_src - tf.reduce_mean(h_src, axis=0) 
	h_trg = h_trg - tf.reduce_mean(h_trg, axis=0 )
	cov_source = (1./(batch_size-1)) * tf.matmul( h_src, h_src, transpose_a=True) #+ gamma * tf.eye(self.hidden_repr_size)
	cov_target = (1./(batch_size-1)) * tf.matmul( h_trg, h_trg, transpose_a=True) #+ gamma * tf.eye(self.hidden_repr_size)
	# Returns the Frobenius norm (there is an extra 1/4 in D-Coral actually)
	# The reduce_mean account for the factor 1/d^2
	return tf.reduce_mean(tf.square( tf.subtract(cov_source,cov_target) )) 

    
    def log_coral_loss(self, h_src, h_trg, gamma=1e-3):
	# regularized covariances result in inf or nan
	# First: subtract the mean from the data matrix
	batch_size = tf.to_float(tf.shape(h_src)[0])
	h_src = h_src - tf.reduce_mean(h_src, axis=0) 
	h_trg = h_trg - tf.reduce_mean(h_trg, axis=0 )
	cov_source = (1./(batch_size-1)) * tf.matmul( h_src, h_src, transpose_a=True) #+ gamma * tf.eye(self.hidden_repr_size)
	cov_target = (1./(batch_size-1)) * tf.matmul( h_trg, h_trg, transpose_a=True) #+ gamma * tf.eye(self.hidden_repr_size)
	#eigen decomposition
	eig_source  = tf.self_adjoint_eig(cov_source)
	eig_target  = tf.self_adjoint_eig(cov_target)
	log_cov_source = tf.matmul( eig_source[1] ,  tf.matmul(tf.diag( tf.log(eig_source[0]) ), eig_source[1], transpose_b=True) )
	log_cov_target = tf.matmul( eig_target[1] ,  tf.matmul(tf.diag( tf.log(eig_target[0]) ), eig_target[1], transpose_b=True) )

	# Returns the Frobenius norm
	return tf.reduce_mean(tf.square( tf.subtract(log_cov_source,log_cov_target))) 
	#~ return tf.reduce_mean(tf.reduce_max(eig_target[0]))
	#~ return tf.to_float(tf.equal(tf.count_nonzero(h_src), tf.count_nonzero(h_src)))

		    
    def build_model(self):
	
        if self.mode == 'train':
	    self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
	    self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
	    self.src_labels = tf.placeholder(tf.int64, [None], 'svhn_labels')
	    
	    self.src_hidden = self.E(self.src_images, is_training = True)
	    self.trg_hidden = self.E(self.trg_images, is_training = True, reuse=True)
	    
	    # last fc layer to logits
	    self.src_logits = self.logits(self.src_hidden)
	    self.trg_logits = self.logits(self.trg_hidden,reuse=True)
	    
	    # class predictions
	    self.src_pred = tf.argmax(self.src_logits, 1)
	    self.src_correct_pred = tf.equal(self.src_pred, self.src_labels)
	    self.src_accuracy = tf.reduce_mean(tf.cast(self.src_correct_pred, tf.float32))

	    # losses: class, domain, total
	    self.class_loss = slim.losses.sparse_softmax_cross_entropy(self.src_logits, self.src_labels)
	    
	    
	    self.trg_softmax = slim.softmax(self.trg_logits)
	    self.trg_entropy = -tf.reduce_mean(tf.reduce_sum(self.trg_softmax * tf.log(self.trg_softmax), axis=1))
	    
	    if self.method == 'log-d-coral':
		print('----------------')
		print('| log-d-coral', self.alpha)
		print('----------------')
		self.domain_loss = self.alpha * self.log_coral_loss(self.src_hidden, self.trg_hidden)
		self.loss = self.class_loss + self.domain_loss
		
	    elif self.method == 'd-coral':
		print('----------------')
		print('| d-coral', self.alpha)
		print('----------------')
		self.domain_loss = self.alpha * self.coral_loss(self.src_hidden, self.trg_hidden)
		self.loss = self.class_loss + self.domain_loss
		
	    elif self.method == 'baseline':
		print('----------------')
		print('| baseline')
		print('----------------')
		self.domain_loss = self.alpha * self.coral_loss(self.src_hidden, self.trg_hidden)
		self.loss = self.class_loss
		
	    elif self.method == 'entropy':
		print('----------------')
		print('| entropy', self.alpha)
		print('----------------')
		self.domain_loss = self.alpha * self.trg_entropy
		self.loss = self.class_loss + self.domain_loss
	    else:
		print('Unrecognized method')
	    
	    self.optimizer = tf.train.AdamOptimizer(self.learning_rate) 
	    self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)
	    
	    # summary op
	    class_loss_summary = tf.summary.scalar('classification_loss', self.class_loss)
	    domain_loss_summary = tf.summary.scalar('domain_loss', self.domain_loss)
	    src_accuracy_summary = tf.summary.scalar('src_accuracy', self.src_accuracy)
	    trg_entropy_summary = tf.summary.scalar('trg_entropy', self.trg_entropy)
	    self.summary_op = tf.summary.merge([class_loss_summary, \
						domain_loss_summary, \
						src_accuracy_summary,\
						trg_entropy_summary])
	
	elif self.mode == 'test':
	    self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
	    self.trg_labels = tf.placeholder(tf.int64, [None], 'mnist_labels')
	    
	    self.trg_hidden = self.E(self.trg_images, is_training = False)
	    
	    # last fc layer to logits
	    self.trg_logits = self.logits(self.trg_hidden)
	    self.trg_softmax = slim.softmax(self.trg_logits)
	    self.trg_entropy = -tf.reduce_mean(tf.reduce_sum(self.trg_softmax * tf.log(self.trg_softmax), axis=1))
	    	    
	    self.trg_pred = tf.argmax(self.trg_logits, 1)
	    self.trg_correct_pred = tf.equal(self.trg_pred, self.trg_labels)
	    self.trg_accuracy = tf.reduce_mean(tf.cast(self.trg_correct_pred, tf.float32))
	
	elif self.mode == 'tsne':
	    self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'mnist_images')
	    self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 3], 'svhn_images')
	    
	    self.trg_hidden = self.E(self.trg_images, is_training = False)
	    self.src_hidden = self.E(self.src_images, is_training = False, reuse = True)
	    
	else:
	    print('Unrecognized mode')
    
