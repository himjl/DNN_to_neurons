import sklearn.model_selection
import numpy as np 
import pandas as pd 

import milsss

df_meta = milsss.get_df_meta('MutatorB0')
choices = df_meta.groupby('obj').apply(lambda mf: mf.iloc[np.random.choice(len(mf), size = 8, replace = False)])['id']
image_ids = list(choices)
obj_labels = choices.index.get_level_values(0)

train_ids, test_ids, train_labels, test_labels = sklearn.model_selection.train_test_split(image_ids, obj_labels, stratify = obj_labels, train_size = 0.5)


import ModelTurk
import sklearn, sklearn.decomposition

gen = ModelTurk.get_off_the_shelf_VGG_19_generator()
