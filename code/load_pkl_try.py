import cPickle as pickle
import numpy as np
import re
from glob import glob
from os.path import join

w, h = (640, 480)

femaler=open('models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
female=pickle.load(femaler)

maler=open('models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
male=pickle.load(maler)

ner=open('models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
n=pickle.load(ner)

'''
testr=open('pkl/image5_smpl.pkl')
test=pickle.load(testr)
female['betas']=test['betas']
female['pose']=test['pose']
female['cam_t']=test['cam_t']
female['cam_f']=test['cam_f']
female['cam_rt']=test['cam_rt']
female['cam_c']=test['cam_c']	
'''

files=[]
pkls=[]
pkl_paths = sorted(glob(join("pkl/","image*[0-9]_smpl_try.pkl")))
#pkl_paths = sorted(glob(join("pkl/","image45_smpl.pkl")))
for ind, pkl_path in enumerate(pkl_paths):
	ind=int(re.findall(r"pkl/image(.+?)_smpl_try.pkl",pkl_path)[0])
	
	pklr=open(pkl_path)
	pkl=pickle.load(pklr)
	female['betas']=pkl['betas']
	female['pose']=pkl['pose']
	female['cam_t']=pkl['cam_t']
	female['cam_f']=pkl['cam_f']
	female['cam_rt']=pkl['cam_rt']
	female['cam_c']=pkl['cam_c']
	with open(pkl_path,'w') as f:
		pickle.dump(female, f, pickle.HIGHEST_PROTOCOL)
	'''
	modify_path="pkl/modify_"+pkl_path[4:];
	with open(modify_path,'w') as f:
		pickle.dump(female, f, pickle.HIGHEST_PROTOCOL)
	print "deforming "+pkl_path+" to "+modify_path;
	'''


