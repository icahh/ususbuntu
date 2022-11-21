import os
import numpy as np
import pandas as pd
class NaiveBayes:
def __init__(self):
self.features = list
self.likelihoods = {}
self.class_priors = {}

self.X_train = np.array
self.y_train = np.array
self.train_size = int
self.num_feats = int
self.debug = True
def fit(self, X, y):
self.features = list(X.columns)
self.X_train = X
self.y_train = y
self.train_size = X.shape[0]
self.num_feats = X.shape[1]
#setting semua nilai likelihood dan prior menjadi 0
self._init_value()
#step 1. hitung likelihood dan prior
self._hitung_likelihood()
self._hitung_class_prior()
def _init_value(self):
for feature in self.features:
self.likelihoods[feature] = {}
for feat_val in np.unique(self.X_train[feature]):
for outcome in np.unique(self.y_train):
self.likelihoods[feature]
.update({feat_val+'_'+outcome: 0})
self.class_priors.update({outcome: 0})
def _hitung_class_prior(self):
for outcome in np.unique(self.y_train):
outcome_count = sum(self.y_train == outcome)
if self.debug:
print("Prior (", outcome, ") = ",
outcome_count, "/", self.train_size)
self.class_priors[outcome]
= outcome_count / self.train_size
def _hitung_likelihood(self):
for feature in self.features:
for outcome in np.unique(self.y_train):
outcome_count = sum(self.y_train == outcome)
feat_likelihood = self.X_train[feature]
[self.y_train[self.y_train == outcome]
.index.values.tolist()].value_counts().to_dict()
for feat_val, count in feat_likelihood.items():
if (self.debug):
print('Likelihood(',feature, '=', feat_val,
'|', outcome, ") :",
count, '/', outcome_count)
self.likelihoods[feature]
[feat_val + '_' + outcome]
= count/outcome_count
def predict(self, X):
results = []
X = np.array(X)
for query in X:
probs_outcome = {}
for outcome in np.unique(self.y_train):
prior = self.class_priors[outcome]
likelihood = 1
for feat, feat_val in zip(self.features, query):
likelihood *= self.likelihoods[feat]
[feat_val + '_' + outcome]
posterior = (likelihood * prior)
probs_outcome[outcome] = posterior
if self.debug:
print("Posterior(", outcome, "|",query,") :"
, posterior)
result = max(probs_outcome,
key=lambda x: probs_outcome[x])
results.append(result)
return np.array(results)