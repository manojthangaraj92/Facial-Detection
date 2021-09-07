#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json 

class get_json:
    def __init__(self, box, probs, landmark):
        self.box = box
        self.probs = probs
        self.landmark = landmark
        
        assert len(self.box) == len(self.probs), f'length of {len(self.box)} is different from {len(self.probs)}'
        assert len(self.box) == len(self.landmark), f'length of {len(self.box)} is different from {len(self.landmark)}'
        
    def get_dict(self, indices):             
        json_dict = dict()
        keypoints = dict()
        json_dict['box'] = self.box[indices]
        json_dict['confidence'] =self.probs[indices]
        keypoints['nose'] = self.landmark[indices][0]
        keypoints['mouth_right'] =self.landmark[indices][1]
        keypoints['right_eye'] = self.landmark[indices][2]
        keypoints['left_eye'] = self.landmark[indices][3]
        keypoints['mouth_left'] = self.landmark[indices][4]
        json_dict['landmarks'] = keypoints
        return json_dict
    
    def return_result(self): 
        lists=list()
        for i in range(len(self.box)):
            x = self.get_dict(i)
            i+=1
            lists.append(x)
        obj = json.dumps(lists)
        return obj 
        
         

