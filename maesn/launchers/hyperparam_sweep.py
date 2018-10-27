from rllab.misc.instrument import VariantGenerator, variant

class VG(VariantGenerator):
  
    @variant
    def seed(self):
        return [x for x in range(2)]

    @variant
    def fast_batch_size(self):
        return [20 , 50]
        #performance is better with 50
    @variant
    def fast_learning_rate(self):
        #For LSBaseline
        #return [0]

        #For Maesn
        return [0.5, 1]
    
    @variant
    def meta_batch_size(self):
        return [20]

    @variant
    def meta_learning_rate(self):
        return [0.01]

    @variant
    def kl_weighting(self):
        return [0.1 , 0.5]
        #return [0 , 0.01, 0.05 , 0.1]

    @variant
    def latent_dim(self):
        return [2]

    @variant
    def init_std(self):
        return [1]

    @variant
    def exp_name(self , fast_batch_size , fast_learning_rate , meta_batch_size , meta_learning_rate , kl_weighting , latent_dim , init_std , seed):
        yield  'fbs_'+str(fast_batch_size)+'_flr_'+str(fast_learning_rate)+'_mbs_'+str(meta_batch_size)+'_mlr_'+str(meta_learning_rate)+'_kl_'+str(kl_weighting)+ \
        '_ldim_'+str(latent_dim)+'_initStd_'+str(init_std)+'_seed_'+str(seed)
