def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01

def cond_probs_product(table, evidence_row, target, target_value):
  evidence_complete = up_zip_lists(up_list_column_names(table)[:-1],evidence_row)
  cond_prob_list = [cond_prob(table, x[0], x[1], target, target_value) for x in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(table,target,target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  #compute P(target=0|...) by using cond_probs_product, take the product of the list, finally multiply by P(target=0) using prior_prob
  target0 = cond_probs_product(table,evidence_row,target,0)*prior_prob(table,target,0)

  #do same for P(target=1|...)
  target1 = cond_probs_product(table,evidence_row,target,1)*prior_prob(table,target,1)

  #Use compute_probs to get 2 probabilities
  [neg,pos] = compute_probs(target0,target1)
  #return your 2 results in a list
  return [neg, pos]

def metrics(zipped_list):
  #asserts here
  assert isinstance(zipped_list, list), f'zipped_list is not a list. It is {type(zipped_list)}.'
  assert all([isinstance(col, list) for col in zipped_list]), f'zipped_list is not a list of lists. {zipped_list=}.'
  assert all((isinstance(item, (list, tuple)) and len(item) == 2) for item in zipped_list), f'zipped_list is not a list of pairs. {zipped_list=}.'
  assert all(all(isinstance(val, int) and val >= 0 for val in item) for item in zipped_list), f'Each value in the pairs must be a non-negative integer. {zipped_list=}.'
  #body of function below
  #first compute the sum of all 4 cases. See code above
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  #now can compute precicision, recall, f1, accuracy. Watch for divide by 0.
  precision = 0 if tp+fp == 0 else tp/(tp+fp) 
  recall = 0 if tp+fn == 0 else tp/(tp+fn) 
  f1 = 0 if precision + recall == 0 else 2 * (precision * recall)/(precision + recall)
  accuracy = sum([1 if pair[0]==pair[1] else 0 for pair in zipped_list])/len(zipped_list)
  #now build dictionary with the 4 measures
  metrics_dict = {'Precision' : precision,
                  'Recall' : recall,
                  'F1' : f1,
                  'Accuracy' : accuracy}
  #finally, return the dictionary
  return metrics_dict 

