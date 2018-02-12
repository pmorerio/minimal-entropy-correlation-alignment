import tensorflow as tf
from model import logDcoral
from solver import Solver

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', "'train', or 'test'")
flags.DEFINE_string('method', 'baseline', "the regularizer: 'baseline' (no regularizer), 'd-coral', 'log-d-coral' or 'entropy'")
flags.DEFINE_string('model_save_path', 'model', "base directory for saving the models")
flags.DEFINE_string('device', '/gpu:0', "/gpu:id number")
flags.DEFINE_string('alpha', '1.', "regularizer weigth")
FLAGS = flags.FLAGS

def main(_):
    
    with tf.device(FLAGS.device):
	model_save_path = FLAGS.model_save_path + '/' + FLAGS.method + '/alpha_' + FLAGS.alpha
	log_dir = 'logs/' + FLAGS.method + '/alpha_' + FLAGS.alpha
	model = logDcoral(mode=FLAGS.mode, method=FLAGS.method, hidden_size = 64, learning_rate=0.0001, alpha=float(FLAGS.alpha))
	solver = Solver(model, model_save_path=model_save_path, log_dir=log_dir)
	
	# create directory if it does not exist
	if not tf.gfile.Exists(model_save_path):
		tf.gfile.MakeDirs(model_save_path)
	
	if FLAGS.mode == 'train':
		solver.train()
	elif FLAGS.mode == 'test':
		solver.test()
	elif FLAGS.mode == 'tsne':
		solver.tsne()
	else:
	    print 'Unrecognized mode.'
        
if __name__ == '__main__':
    tf.app.run()



    


