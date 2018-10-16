

import csv
from matplotlib import pyplot as plt

def plot(_file, num_itr = 200):

    origHis = []
       
            
    with open(_file+"/progress.csv", 'r') as f:
        reader = csv.reader(f, delimiter=',')
        
        row = None
        records = []
        for row in reader:
            records.append(row)
        ret1Ind = 0 
        for i in range(len(records[0])):
            if records[0][i]=="1AverageReturn":
                ret1Ind = i

        import ipdb
        ipdb.set_trace()
           
        
        ret1 = []
        for record in records[1:]:
            ret1.append(float(record[ret1Ind]))
            #ret1.append(record[ret1Ind])
       
        return ret1[:num_itr]
            

plot('/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/nipsRebuttal-masen-wheeledRobot-radius2-train100-subsampling/fulltrpomasen1_ldim_2_fbs5_mbs20_flr_1metalr_0.01_step11kl_schemeNonekl_weighting0.5')