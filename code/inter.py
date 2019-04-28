import cPickle as pickle
testr=open('test.pkl')
test=pickle.load(testr)
x=test['pose']

prer=open('pre.pkl')
pre=pickle.load(prer)
y=pre['pose']

z=(x+y)/2

mainr=open('main.pkl')
main=pickle.load(mainr)
main['pose']=z

with open('result_test.pkl','w') as f:
    pickle.dump(main, f, pickle.HIGHEST_PROTOCOL)

