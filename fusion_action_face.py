import torch
import pickle
import numpy as np
import tensorflow as tf

def valid():
	stgcn_file='test_result.pkl'
	EmotionFAN_file='self-attention_1_91.954'

	allsum = np.zeros([174,2])
	labels = []
	
	#Emotion-FAN
	FAN_pred=torch.load(EmotionFAN_file)
	FAN_scoretensor=FAN_pred['prec_score']
	FAN_labeltensor=FAN_pred['target_vector']

	#session = tf.compat.v1.Session()
	FAN_score = FAN_scoretensor.cpu().numpy()
	FAN_label = FAN_labeltensor.cpu().numpy()
	labels=FAN_label
	print("fan_score:",FAN_score)
	print("fan:",FAN_score.shape)
	
	#st-gcn
	gcn_score = np.empty([174,2])
	gcn_pred=pickle.load(open(stgcn_file,"rb"))
	for i,key in zip(range(174),gcn_pred):
		gcn_score = np.insert(gcn_score, i, values=gcn_pred[key], axis=0)
	gcn_score=gcn_score[0:174,:]
	print("gcn_score:",gcn_score)
	print("gcn:",gcn_score.shape)
		
	allsum=FAN_score+gcn_score
	print("allsum:",allsum)
	print("usr_video_score:",allsum[173])
	if(allsum[173][0]>allsum[173][1]):
		print("usr_video_prediction: not listening carefully")
	else:
		print("usr_video_prediction: listening carefully")
	preds = np.argmax(allsum,axis=1)

	num_correct = np.sum(preds == labels)
	acc = num_correct * 1.0 / preds.shape[0]
	print('acc=%.3f' % (acc))



if __name__ == '__main__':
	valid()
	valid()
