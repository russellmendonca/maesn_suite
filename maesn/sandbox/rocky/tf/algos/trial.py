import joblib
import tensorflow as tf

with tf.Session() as sess:
    loaded = joblib.load(_file)

   
    self.policy = loaded['policy']
    fobj = open('params.pkl', 'wb')

    pickle.dump(self.policy.get_param_values(), fobj)
    fobj.close()
            
