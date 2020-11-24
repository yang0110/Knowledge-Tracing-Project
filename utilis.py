import numpy as np 
import pandas as pd 


def student_ability(train_data, user_id):
	student_data = train_data[train_data.user_id.values==user_id]
	student_data=student_data[student_data.content_id.values==0]
	total_question = student_data.shape[0]
	correct_count = student_data[student_data.correct==1].shape[0]
	ability = correct_count/total_question
	return ability, correct_count, total_question

def update_ability(new_data, correct_count, total_question) :
	if new_data.correct == 1:
		correct_count +=1
		total_question +=1
	ability = correct_count/total_question
	return ability

def question_hardness(train_data, ques_id):
	ques_data = train_data[train_data.context_type==0]
	ques_data = ques_data[ques_data.content_id==ques_id]
	correct_count = ques_data[ques_data.correct == 1].shape[0]
	total = ques_data.shape[0]
	hardness = correct_count/total 
	return hardness

def update_hardness(new_data, correct_count, total):
	if new_data.correct == 1:
		correct_count += 1
		total += 1
	hardness = correct_count/total
	return hardness
