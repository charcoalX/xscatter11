from utils import util_import_error_message

# psycopg2
try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    raise util_import_error_message(['psycopg2'])

# waitress, flask
try:
    #from waitress import serve
    from flask import Flask, render_template, request, jsonify
except ImportError:
    raise util_import_error_message(['waitress', 'flask'])

# json
try:
    import json
except ImportError:
    raise util_import_error_message(['json'])

# sys
try:
    import sys
except ImportError:
    raise util_import_error_message(['sys'])

# subprocess
try:
    import subprocess
except ImportError:
    raise util_import_error_message(['subprocess'])

# os
try:
    import os
except ImportError:
    raise util_import_error_message(['os'])

# sklearn.metrics
try:
    from sklearn.metrics import confusion_matrix
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn import metrics
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.metrics import davies_bouldin_score
    from scipy import stats
    # tsne
    from sklearn.manifold import TSNE
    # AP measure
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    #from inspect import signature

except ImportError:
    raise util_import_error_message(['sklearn','matplotlib','inspect'])

# numpy
try:
    import numpy
except ImportError:
    raise util_import_error_message(['numpy'])

#import pyitlib
try:
    # from pyitlib import discrete_random_variable as drv
    import math
except ImportError:
    raise util_import_error_message(['math'])

# requests (for LRP sidecar proxy)
try:
    import requests as http_requests
except ImportError:
    raise util_import_error_message(['requests'])