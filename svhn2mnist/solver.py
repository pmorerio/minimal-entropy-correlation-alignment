import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import time

import matplotlib.pyplot as plt
import matplotlib as mpl

import utils
from sklearn.manifold import TSNE
from scipy import misc

#~ from utils import resize_images

class Solver(object):

    def __init__(self, model, batch_size=128, train_iter=100000, 
                 svhn_dir='svhn',  mnist_dir='mnist', log_dir='logs',
		 model_save_path='model', trained_model='model/model'):
        
        self.model = model
        self.batch_size = batch_size
        self.train_iter = train_iter
        self.svhn_dir = svhn_dir
	self.mnist_dir = mnist_dir
        self.log_dir = log_dir
        self.model_save_path = model_save_path
        self.trained_model = model_save_path + '/model'
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True


    def load_mnist(self, image_dir, split='train'):
        print ('Loading MNIST dataset.')
	
	image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
	
        return images, np.squeeze(labels).astype(int)

    def load_svhn(self, image_dir, split='train'):
        print ('Loading SVHN dataset.')
        
        image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'
            
        image_dir = os.path.join(image_dir, image_file)
        svhn = scipy.io.loadmat(image_dir)
        images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
	#~ images= resize_images(images)
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels==10)] = 0
        return images, labels
	
    def train(self):
	
	# make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
	
	print 'Training.'
	
	trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='train')
	trg_test_images, trg_test_labels = self.load_mnist(self.mnist_dir, split='test')

	src_images, src_labels = self.load_svhn(self.svhn_dir, split='train')
	src_test_images, src_test_labels = self.load_svhn(self.svhn_dir, split='test')

        # build a graph
        model = self.model
        model.build_model()
	
	config = tf.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
	   
	    print ('Start training.')
	    trg_count = 0
	    t = 0
	    start_time = time.time()
	    for step in range(self.train_iter):
		
		trg_count += 1
		t+=1
		
		i = step % int(src_images.shape[0] / self.batch_size)
		j = step % int(trg_images.shape[0] / self.batch_size)
		       
		feed_dict = {model.src_images: src_images[i*self.batch_size:(i+1)*self.batch_size], 
				model.src_labels: src_labels[i*self.batch_size:(i+1)*self.batch_size], 
				model.trg_images: trg_images[j*self.batch_size:(j+1)*self.batch_size],
				}  
		
		sess.run(model.train_op, feed_dict) 

		if t%5000==0 or t==1:

		    summary, l_c, l_d, src_acc = sess.run([model.summary_op, model.class_loss, model.domain_loss, model.src_accuracy], feed_dict)
		    summary_writer.add_summary(summary, t)
		    print ('Step: [%d/%d]  c_loss: [%.6f]  d_loss: [%.6f]  train acc: [%.2f]' \
			       %(t, self.train_iter, l_c, l_d, src_acc))
		  
		#~ if t%10000==0:
		    #~ print 'Saved.'
	    with open('time_' + str(model.alpha) +'_'+ model.method + '.txt', "a") as resfile:
		resfile.write(str( (time.time()-start_time)/float(self.train_iter) )+'\n')
	    saver.save(sess, os.path.join(self.model_save_path, 'model'))
	    
    def test(self):
	
	trg_images, trg_labels = self.load_mnist(self.mnist_dir, split='test')       
	
	# build a graph
	model = self.model
	model.build_model()
		
	config = tf.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	
        with tf.Session(config=config) as sess:
	    tf.global_variables_initializer().run()
		
	    print ('Loading  model.')
	    variables_to_restore = slim.get_model_variables()
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.trained_model)
    

	    trg_acc, trg_entr = sess.run(fetches=[model.trg_accuracy, model.trg_entropy], 
					    feed_dict={model.trg_images: trg_images[:], 
							model.trg_labels: trg_labels[:]})
					      
	    print ('test acc [%.3f]' %(trg_acc))
	    print ('entropy [%.3f]' %(trg_entr))
	    with open('test_'+ str(model.alpha) +'_' + model.method + '.txt', "a") as resfile:
		resfile.write(str(trg_acc)+'\t'+str(trg_entr)+'\n')
	    
	    #~ print confusion_matrix(trg_labels, trg_pred)
	    
    def tsne(self, n_samples = 2000):
	
	source_images, source_labels =  self.load_svhn(self.svhn_dir, split='test')
	target_images, target_labels =  self.load_mnist(self.mnist_dir, split='test')
	
	model = self.model
	model.build_model()
	
	config = tf.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	
        with tf.Session(config=config) as sess:
	    print ('Loading test model.')
	    variables_to_restore = tf.global_variables() 
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, self.trained_model)
	
	    target_images = target_images[:n_samples]
	    target_labels = target_labels[:n_samples]
	    source_images = source_images[:n_samples]
	    source_labels = source_labels[:n_samples]
	    print(source_labels.shape)
	    
	    assert len(target_labels) == len(source_labels)
	    
	    src_labels = utils.one_hot(source_labels.astype(int),10 )
	    trg_labels = utils.one_hot(target_labels.astype(int),10 )

	    n_slices = int(n_samples/self.batch_size)
	    
	    fx_src = np.empty((0,model.hidden_repr_size))
	    fx_trg = np.empty((0,model.hidden_repr_size))
	    
	    for src_im, trg_im in zip(np.array_split(source_images, n_slices),  
					np.array_split(target_images, n_slices),  
					):
								    
		feed_dict = {model.src_images: src_im, model.trg_images: trg_im}
		
		fx_src_, fx_trg_ = sess.run([model.src_hidden, model.trg_hidden], feed_dict)
		
		
		fx_src = np.vstack((fx_src, np.squeeze(fx_src_)) )
		fx_trg = np.vstack((fx_trg, np.squeeze(fx_trg_)) )
	    
	    src_labels = np.argmax(src_labels,1)
	    trg_labels = np.argmax(trg_labels,1)
	    
	    assert len(src_labels) == len(fx_src)
	    assert len(trg_labels) == len(fx_trg)
	    
	    print 'Computing T-SNE.'

	    model = TSNE(n_components=2, random_state=0)
	    
	    
	    TSNE_hA = model.fit_transform(np.vstack((fx_src,fx_trg)))
	    plt.figure(2)
	    plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((src_labels, trg_labels, )), s=3,  cmap = mpl.cm.jet)
	    plt.figure(3)
	    plt.scatter(TSNE_hA[:,0], TSNE_hA[:,1], c = np.hstack((np.ones((n_samples,)), 2 * np.ones((n_samples,)))), s=3,  cmap = mpl.cm.jet)
		
	    plt.show()
	    
		    
if __name__=='__main__':

    print('empty')
