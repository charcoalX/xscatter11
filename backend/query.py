from utils import util_compute_distance
from modules import *
from info import *

dataSize  = 500

def query_all(params, conn, cursor):
    """ Query cifar10 or x-ray data """

    # Parse params before query
    data_type = params['Data type']
    # feature_vector = params['Vector of feature']
    embedding_method = params['Embedding method']
    error_distance = params['Distance of error']

    # Create new table name
    table_name = data_type + '_' + embedding_method.replace(' ', '')
    print("data_type:",data_type)
    print("table_name:",table_name)
    # Execute query
    conn.rollback()  # clear any prior failed transaction
    if data_type == 'cifar10':
        cursor.execute('SELECT * from ' + table_name + '_1000 limit 300')
    else:
        cursor.execute('SELECT * from ' + table_name +  '_'+ str(dataSize)+'   limit 300')
        #cursor.execute('SELECT * from ' + table_name + '_1000_goodiui  limit 300')   # _6861 and limit 500  is for vis case3   # In xscatter8 _1000 limit 500
        #cursor.execute('SELECT * from ' + table_name + '_1000 limit 300')   # _6861 and limit 500  is for vis case3   # In xscatter8 _1000 limit 500

    # Generate query result
    query_result = {}
    # Create rows
    records = cursor.fetchall()
    
    for row in records:
        ID = row[0]
        query_result[ID] = {}
        query_result[ID]['feature_vector'] = row[1]
        query_result[ID]['feature2048-x'] = row[2]
        query_result[ID]['feature2048-y'] = row[3]
        query_result[ID]['prediction17-x'] = row[4]
        query_result[ID]['prediction17-y'] = row[5]
        query_result[ID]['truelabel17-x'] = row[6]
        query_result[ID]['truelabel17-y'] = row[7]
        # Relu
        query_result[ID]["s1-x"] = row[8]
        query_result[ID]["s1-y"] = row[9]
        query_result[ID]["s2-x"] = row[10]
        query_result[ID]["s2-y"] = row[11]
        query_result[ID]["s3-x"] = row[12]
        query_result[ID]["s3-y"] = row[13]
        query_result[ID]["s4-x"] = row[14]
        query_result[ID]["s4-y"] = row[15]
        query_result[ID]["s5-x"] = row[16]
        query_result[ID]["s5-y"] = row[17]

        # Create true label and prediction prob
        if data_type == 'cifar10':
            col_start = 20
            col_end = 40
        else:
            col_start = 8
            col_end = 42
        
        true_label = [row[x] for x in range(col_start, col_end) if x % 2 == 0]
        pred_prob = [row[x] for x in range(col_start, col_end) if x % 2 == 1]

        query_result[ID]['trueLabel'] = true_label
        query_result[ID]['predProb'] = pred_prob
        query_result[ID]['Distance of error'] = util_compute_distance(true_label, pred_prob, error_distance)
    
    return query_result

def query_confusion_matrix(params, conn, cursor):
    """ Columns query for cifar10 as well as x-ray data for confusion matrix """
    print('Query confusion matrix: ', params)

def query_clustering(params):
    """ K-Means clustering """

    # Get all parameters
    cluster_number = int(params['clusterNum'])
    vectors = params['vectors']

    # Clustering
    query_result = {}
    cluster = KMeans(n_clusters = cluster_number, random_state=0).fit(vectors)
    query_result['cluster'] = cluster.labels_.tolist()

    ########################################
   
    davies_score = davies_bouldin_score(vectors, cluster.labels_)
    Silh_score = metrics.silhouette_score(vectors, cluster.labels_)

    query_result['Silh_score'] = Silh_score
    query_result['davies_score'] = davies_score
    
    # ############################################
    return query_result



def query_clustering_DBSCAN(params):
    """ DBSCAN clustering """
    # Get all parameters
    eps = float(params['eps'])
    min_samples = int(params['min_samples'])
    vectors = params['vectors']
    # Clustering
    query_result = {}
    cluster = DBSCAN(eps, min_samples).fit(vectors)
    labels = cluster.labels_.tolist()
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
   
    # print("cluster.labels_:",n_clusters_, " , labels list:",labels)

    if  n_clusters_ >1 :
        davies_score = davies_bouldin_score(vectors, cluster.labels_)
        Silh_score = metrics.silhouette_score(vectors, cluster.labels_)
    else:
        print("single cluster")
        davies_score = 100
        Silh_score = 100
  
    query_result['Silh_score'] = Silh_score
    query_result['davies_score'] = davies_score
    query_result['num_cluster'] = len(set(labels))

    #### make the outliers to be a cluster at the index 0, this one also work but outliers will be listed at the top 
    # query_result['cluster'] = [i+1 for i in labels ] if -1 in labels else labels

    #### make the outliers to be a cluster at the last index 
    if -1 not in labels:
        query_result['cluster'] = labels
        query_result['outlier'] = False
    else:
        new_labels = []
        for i in labels:
            if i == -1:
                new_labels.append(i+query_result['num_cluster'])
            else:
                new_labels.append(i)
        query_result['cluster'] = new_labels
        query_result['outlier'] = True

    # ############################################
    return query_result

    
def query_get_mutual_info(params, conn, cursor):

    data_type = params['Data type']
    embedding_method = params['Embedding method']
    table_name = data_type + '_' + embedding_method.replace(' ', '')
    # method = params['matrix method']

    query_str_truelabel = ''
    query_str_predprob = ''
    n_classes = 10

    if data_type == 'cifar10':
        n_classes = 10
    else:
        n_classes = 17
    for i in range(1, n_classes+1):
        query_str_truelabel += 'truelabel_c' + str(i) + ','
        query_str_predprob += 'predprob_c' + str(i) + ','

    query_str_truelabel = query_str_truelabel[:-1]
    query_str_predprob = query_str_predprob[:-1]

    # True label
    conn.rollback()  # clear any prior failed transaction
    cursor.execute("SELECT " + query_str_truelabel + " FROM " + table_name +  "_"+ str(dataSize)+"   limit 1000")  # _5000 and limit 5000  is for vis case3  # In xscatter8 _1000 limit 1000

    # cursor.execute("SELECT " + query_str_truelabel + " FROM " + table_name + "_1000 limit 1000")  # _5000 and limit 5000  is for vis case3  # In xscatter8 _1000 limit 1000
    truelabel_vector = cursor.fetchall()
    truelabel_vector = numpy.array(truelabel_vector)

    # Process transapose matrix
    reshape_truelabel_vector = truelabel_vector.T

    # Predprob
    cursor.execute("SELECT " + query_str_predprob + " FROM " + table_name +  "_"+ str(dataSize)+"  limit 1000") # _5000 and limit 5000  is for vis case3  # In xscatter8 _1000 limit 1000

    # cursor.execute("SELECT " + query_str_predprob + " FROM " + table_name + "_1000 limit 1000") # _5000 and limit 5000  is for vis case3  # In xscatter8 _1000 limit 1000
    predprob_vector = cursor.fetchall()
    predprob_vector = numpy.array(predprob_vector)
    predprob_vector = numpy.where(predprob_vector > 0.5, 1, 0)

    # Process transpose matrix
    reshape_predprob_vector = predprob_vector.T

    # Compute mutual information
    recommendation = {}    
   
    
    methods = ['MI', 'correlation','conditional_entropy']
    for method in methods:
        print("method",method)
        mutual_info = {}
        mutual_info['trueLabel'] = {}
        mutual_info['predProb'] = {}
        mutual_info['between'] = {}

        for i in range(n_classes):
           
            truelabel_attr1 = reshape_truelabel_vector[i]
            predprob_attr1 = reshape_predprob_vector[i]

            for j in range(n_classes):
                truelabel_attr2 = reshape_truelabel_vector[j]
                predprob_attr2 = reshape_predprob_vector[j]

                if method == 'MI':

                    # if np.sum(truelabel_attr1) == 0 or  np.sum(truelabel_attr2) == 0:                        
                    #     # res1 = 10000000
                    #     res1 = metrics.adjusted_mutual_info_score(truelabel_attr1, truelabel_attr2)
                    # else:
                    #     res1 = metrics.adjusted_mutual_info_score(truelabel_attr1, truelabel_attr2)
                        
                    # if np.sum(predprob_attr1)  == 0 or np.sum(predprob_attr2) == 0:
                    #     # res2 = 10000000
                    #     res2 = metrics.adjusted_mutual_info_score(predprob_attr1, predprob_attr2)
                    # else:
                    #     res2 = metrics.adjusted_mutual_info_score(predprob_attr1, predprob_attr2)
                        
                    # if np.sum(truelabel_attr1) == 0 or np.sum(predprob_attr2) == 0:
                    #     # res3 = 10000000
                    #     res3 = metrics.adjusted_mutual_info_score(truelabel_attr1, predprob_attr2) 
                    # else:
                    #     res3 = metrics.adjusted_mutual_info_score(truelabel_attr1, predprob_attr2) 
  

                    res1 = metrics.adjusted_mutual_info_score(truelabel_attr1, truelabel_attr2)
                    res2 = metrics.adjusted_mutual_info_score(predprob_attr1, predprob_attr2)
                    res3 = metrics.adjusted_mutual_info_score(truelabel_attr1, predprob_attr2)

                    mutual_info['trueLabel'][str(i) + '-' + str(j)] = res1
                    mutual_info['predProb'][str(i) + '-' + str(j)] = res2
                    mutual_info['between'][str(i) + '-' + str(j)] = res3            

                elif method == 'correlation':
                                        
                    if np.sum(truelabel_attr1) == 0 or  np.sum(truelabel_attr2) == 0:                        
                        res1 = 0 #10000000
                    else:
                        res1 = numpy.corrcoef(truelabel_attr1, truelabel_attr2)[1][0]
                        
                    if np.sum(predprob_attr1)  == 0 or np.sum(predprob_attr2) == 0:
                        res2 = 0 #10000000
                    else:
                        res2 = numpy.corrcoef(predprob_attr1, predprob_attr2)[1][0]
                        
                    if np.sum(truelabel_attr1) == 0 or np.sum(predprob_attr2) == 0:
                        res3 = 0 #10000000
                    else:
                        res3 = numpy.corrcoef(truelabel_attr1, predprob_attr2)[1][0] 
  
                    mutual_info['trueLabel'][str(i) + '-' + str(j)] = float(res1)
                    mutual_info['predProb'][str(i) + '-' + str(j)] = float(res2)
                    mutual_info['between'][str(i) + '-' + str(j)] = float(res3)

                elif method == 'conditional_entropy':
                    # if np.sum(truelabel_attr1) == 0 or  np.sum(truelabel_attr2) == 0:                        
                    #     res1 = 0#10000000
                    # else:
                    #     res1 = conditional_entropy(truelabel_attr1, truelabel_attr2)
                        
                    # if np.sum(predprob_attr1)  == 0 or np.sum(predprob_attr2) == 0:
                    #     res2 = 0#10000000
                    # else:
                    #     res2 = conditional_entropy(predprob_attr1, predprob_attr2)
                        
                    # if np.sum(truelabel_attr1) == 0 or np.sum(predprob_attr2) == 0:
                    #     res3 = 0#10000000
                    # else:
                    #     res3 = conditional_entropy(truelabel_attr1, predprob_attr2)


                    res1 = conditional_entropy(truelabel_attr1, truelabel_attr2)
                    res2 = conditional_entropy(predprob_attr1, predprob_attr2)
                    res3 = conditional_entropy(truelabel_attr1, predprob_attr2)
                    
                    if math.isnan(res1):
                         mutual_info['trueLabel'][str(i) + '-' + str(j)] = 0
                    else:
                        mutual_info['trueLabel'][str(i) + '-' + str(j)] = float(res1)
                    if math.isnan(res2):
                         mutual_info['predProb'][str(i) + '-' + str(j)] = 0
                    else:
                        mutual_info['predProb'][str(i) + '-' + str(j)] = float(res2)
                    if math.isnan(res3):
                         mutual_info['between'][str(i) + '-' + str(j)] = 0
                    else:
                        mutual_info['between'][str(i) + '-' + str(j)] = float(res3)
                
                if (numpy.sum(truelabel_attr1) == 0 and numpy.sum(truelabel_attr2) == 0) \
                    or ( numpy.sum(predprob_attr1) == 0 and numpy.sum(predprob_attr2) ==0 ) \
                    or ( numpy.sum(truelabel_attr1) == 0 and numpy.sum(predprob_attr2) ==0):                               
                    print("i=",i,"j=",j,"numpy.sum(truelabel_attr1)=",numpy.sum(truelabel_attr1),"numpy.sum(truelabel_attr2))=",numpy.sum(truelabel_attr2)),"numpy.sum(predprob_attr1) =",numpy.sum(predprob_attr1),"numpy.sum(predprob_attr2)=",numpy.sum(predprob_attr2)
                                                  
                    mutual_info['trueLabel'][str(i) + '-' + str(j)] = float(10000000)
                    mutual_info['predProb'][str(i) + '-' + str(j)] = float(10000000)
                    mutual_info['between'][str(i) + '-' + str(j)] = float(10000000)

                 

        recommendation[method] = mutual_info
    # print("conditional_entropy")
    # print(recommendation['conditional_entropy'])  
    #################
    # for AP, mAP

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    # n_classes =17
    for i in range(n_classes):
        truelabel_attr1 = reshape_truelabel_vector[i]
        predprob_vector = reshape_predprob_vector[i]
        predprob_attr1 = numpy.where(predprob_vector > 0.5, 1, 0)
        print("attributes i=", i)
        print("truelabel_attr1.shape = ",truelabel_attr1.shape)

        #import pdb
        #pdb.set_trace()
        precision[i], recall[i], _ = precision_recall_curve(truelabel_attr1,predprob_attr1 , pos_label= 1)

        #average_precision[i] = round(average_precision_score(truelabel_attr1,predprob_attr1 )*1000)/1000

        print("precision, recall :",precision[i], recall[i] ,_ )
    #print("average_precision:",average_precision)

    return recommendation




####  count table ### 
def query_get_count_info(params, conn, cursor):

    data_type = params['Data type']
    embedding_method = params['Embedding method']
    table_name = data_type + '_' + embedding_method.replace(' ', '')
    

    query_str_truelabel = ''
    query_str_predprob = ''
    n_classes = 10

    if data_type == 'cifar10':
        n_classes = 10
    else:
        n_classes = 17

    for i in range(1, n_classes +1):
        query_str_truelabel += 'truelabel_c' + str(i) + ','
        query_str_predprob += 'predprob_c' + str(i) + ','

    query_str_truelabel = query_str_truelabel[:-1]
    query_str_predprob = query_str_predprob[:-1]

    # True label
    conn.rollback()  # clear any prior failed transaction
    cursor.execute("SELECT " + query_str_truelabel + " FROM " + table_name + "_"+ str(dataSize)+" limit 300")
    truelabel_vector = cursor.fetchall()
    truelabel_vector = numpy.array(truelabel_vector)

    # Process 1 transapose matrix
    reshape_truelabel_vector = truelabel_vector.T

    # Predprob
    cursor.execute("SELECT " + query_str_predprob + " FROM " + table_name +  "_"+ str(dataSize)+"  limit 300")
    predprob_vector = cursor.fetchall()
    predprob_vector = numpy.array(predprob_vector)
    predprob_vector = numpy.where(predprob_vector > 0.5, 1, 0)

    # Process transpose matrix
    reshape_predprob_vector = predprob_vector.T
    print(reshape_predprob_vector[0].shape)

    # Process 2
    attrNum = reshape_predprob_vector.shape[0]
    arrtibutes_combination = {}  # return obj

    numberOfCombination = 1
    maxNumAttrForGroup = 5

    while numberOfCombination <= maxNumAttrForGroup: 
        counter = {} # each attributes has its own count
        correctedPredictedImageIDs = {}
        correctedPredictedImageIDs2 = {} 
        correctedPredictedImageIDs3 = {} 
        correctedPredictedImageIDs4 = {} 
        correctedPredictedImageIDs5 = {} 
        ###################
        # loop through attributes number, single attribute case
        for attr_i in range(attrNum):
            true_attr1 = reshape_truelabel_vector[attr_i]
            pred_attr1 = reshape_predprob_vector[attr_i]

            # filter the data based on the true_attr1 where ==1 
            true_attr1_filtered_idx = numpy.where(true_attr1 ==1)
            pred_attr1_filtered_idx = numpy.where(pred_attr1 ==1)
            # print("true_attr1_filtered_idx",true_attr1_filtered_idx )
            correctedPredictedImageIDs[str(attr_i)] =[]
            correctedPredictedImageIDs[str(attr_i)] =numpy.intersect1d(true_attr1_filtered_idx,pred_attr1_filtered_idx)
            #print("correctedPredictedImageIDs 1:",numberOfCombination, attr_i, correctedPredictedImageIDs[str(attr_i)])

            ###################
            true_attr1_filtered = true_attr1[true_attr1_filtered_idx]
            pred_attr1_filtered = pred_attr1[true_attr1_filtered_idx]

            if numberOfCombination ==1 and len(true_attr1_filtered_idx) > 0:
                counter['attr_'+str(attr_i)] = {} # TP, TN ...
                counter['attr_'+str(attr_i)]['imageIDs'] = numpy.array(true_attr1_filtered_idx).tolist()
                counter['attr_'+str(attr_i)]['imageIDs_correctPred'] = numpy.array(correctedPredictedImageIDs[str(attr_i)] ).tolist()
                counter['attr_'+str(attr_i)][str(attr_i)] = getAttr_info(attr_i,true_attr1_filtered_idx ,true_attr1_filtered, pred_attr1_filtered)

                # if len(counter['attr_'+str(attr_i)]['imageIDs']) < len(counter['attr_'+str(attr_i)]['imageIDs_correctPred'])
                #     print("issue:",attr_i,len(counter['attr_'+str(attr_i)]['imageIDs']) , len(counter['attr_'+str(attr_i)]['imageIDs_correctPred'])
            

            ###################
            
            # filter the data based on all true_attr1,2, .. where ==1, and update index
            if numberOfCombination > 1:
                for attr_j in range(attr_i +1 ,attrNum):
                    true_attr2 = reshape_truelabel_vector[attr_j]
                    pred_attr2 = reshape_predprob_vector[attr_j]
                    # filter the data based on the true_attr1 where ==1 
                    true_attr2_filtered_idx = numpy.where(true_attr2 ==1)
                    pred_attr2_filtered_idx = numpy.where(pred_attr2 ==1)    
                    tmp =  numpy.intersect1d(true_attr2_filtered_idx,pred_attr2_filtered_idx)   
                    # correctedPredictedImageIDs2 = {}             
                    correctedPredictedImageIDs2[str(attr_i)+'-'+str(attr_j)] = numpy.intersect1d(correctedPredictedImageIDs[str(attr_i)], tmp)
                    true_attr_filtered_idx = numpy.intersect1d(true_attr1_filtered_idx,true_attr2_filtered_idx)


                    true_attr2_filtered = true_attr2[true_attr_filtered_idx]
                    pred_attr2_filtered = pred_attr2[true_attr_filtered_idx]

                    true_attr1_filtered = true_attr1[true_attr_filtered_idx]
                    pred_attr1_filtered = pred_attr1[true_attr_filtered_idx]


                    if numberOfCombination ==2  and len(true_attr_filtered_idx) > 0:
                        counter['attr_'+str(attr_i)+'_'+str(attr_j)] = {}
                        counter['attr_'+str(attr_i)+'_'+str(attr_j)]['imageIDs'] = numpy.array(true_attr_filtered_idx).tolist()
                        counter['attr_'+str(attr_i)+'_'+str(attr_j)]['imageIDs_correctPred'] = numpy.array(correctedPredictedImageIDs2[str(attr_i)+'-'+str(attr_j)]).tolist()
                        counter['attr_'+str(attr_i)+'_'+str(attr_j)][str(attr_i)] = getAttr_info(attr_i, true_attr_filtered_idx,true_attr1_filtered, pred_attr1_filtered)
                        counter['attr_'+str(attr_i)+'_'+str(attr_j)][str(attr_j)] = getAttr_info(attr_j, true_attr_filtered_idx,true_attr2_filtered, pred_attr2_filtered)

                        #  if len(counter['attr_'+str(attr_i)+'_'+str(attr_j)]['imageIDs']) < len(counter['attr_'+str(attr_i)+'_'+str(attr_j)]['imageIDs_correctPred'])
                            # print("issue:",attr_i,len(counter['attr_'+str(attr_i)]['imageIDs']) , len(counter['attr_'+str(attr_i)]['imageIDs_correctPred'])
            


                    if numberOfCombination > 2:
                        for attr_k in range(attr_j +1 ,attrNum):

                            true_attr3 = reshape_truelabel_vector[attr_k]
                            pred_attr3 = reshape_predprob_vector[attr_k]
                            # filter the data based on the true_attr1 where ==1 
                            true_attr3_filtered_idx = numpy.where(true_attr3 ==1)
                            pred_attr3_filtered_idx = numpy.where(pred_attr3 ==1)                    
                           
                            # correctedPredictedImageIDs3 = {}           
                            tmp =numpy.intersect1d(true_attr3_filtered_idx,pred_attr3_filtered_idx)                    
                            correctedPredictedImageIDs3[str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k)] = numpy.intersect1d(tmp, correctedPredictedImageIDs2[str(attr_i)+'-'+str(attr_j)])
                           
                            true_attr_filtered_idx = numpy.intersect1d(true_attr_filtered_idx,true_attr3_filtered_idx)
                            

                            true_attr3_filtered = true_attr3[true_attr_filtered_idx]
                            pred_attr3_filtered = pred_attr3[true_attr_filtered_idx]

                            true_attr2_filtered = true_attr2[true_attr_filtered_idx]
                            pred_attr2_filtered = pred_attr2[true_attr_filtered_idx]

                            true_attr1_filtered = true_attr1[true_attr_filtered_idx]
                            pred_attr1_filtered = pred_attr1[true_attr_filtered_idx]


                            if numberOfCombination ==3  and len(true_attr_filtered_idx) > 0:
                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k)] = {}
                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k)]['imageIDs'] = numpy.array(true_attr_filtered_idx).tolist()
                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k)]['imageIDs_correctPred'] = numpy.array(correctedPredictedImageIDs3[str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k)]).tolist()
                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k)][str(attr_i)] =getAttr_info(attr_i,true_attr_filtered_idx, true_attr1_filtered, pred_attr1_filtered)
                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k)][str(attr_j)] =getAttr_info(attr_j,true_attr_filtered_idx, true_attr2_filtered, pred_attr2_filtered)
                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k)][str(attr_k)] =getAttr_info(attr_k,true_attr_filtered_idx, true_attr3_filtered, pred_attr3_filtered)

                            if numberOfCombination > 3:
                                for attr_m in range(attr_k +1 ,attrNum):

                                    true_attr4 = reshape_truelabel_vector[attr_m]
                                    pred_attr4 = reshape_predprob_vector[attr_m]
                                    # filter the data based on the true_attr1 where ==1 
                                    true_attr4_filtered_idx = numpy.where(true_attr4 ==1)
                                    pred_attr4_filtered_idx = numpy.where(pred_attr4 ==1)      
                                    # correctedPredictedImageIDs4 ={}              
                                    tmp =numpy.intersect1d(true_attr4_filtered_idx,pred_attr4_filtered_idx)
                                    correctedPredictedImageIDs4['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)] = numpy.intersect1d(tmp, correctedPredictedImageIDs3[str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k)])
                                  
                                    # update the index list: true_attr1_filtered_idx
                                    true_attr_filtered_idx = numpy.intersect1d(true_attr_filtered_idx,true_attr4_filtered_idx)

                                    true_attr4_filtered = true_attr4[true_attr_filtered_idx]
                                    pred_attr4_filtered = pred_attr4[true_attr_filtered_idx]

                                    true_attr3_filtered = true_attr3[true_attr_filtered_idx]
                                    pred_attr3_filtered = pred_attr3[true_attr_filtered_idx]

                                    true_attr2_filtered = true_attr2[true_attr_filtered_idx]
                                    pred_attr2_filtered = pred_attr2[true_attr_filtered_idx]

                                    true_attr1_filtered = true_attr1[true_attr_filtered_idx]
                                    pred_attr1_filtered = pred_attr1[true_attr_filtered_idx]


                                    if numberOfCombination ==4  and len(true_attr_filtered_idx) > 0:
                                        counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)] = {}
                                        counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)]['imageIDs'] = numpy.array(true_attr_filtered_idx).tolist()
                                        counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)]['imageIDs_correctPred'] = numpy.array(correctedPredictedImageIDs4['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)]).tolist()
                                        counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)][str(attr_i)] = getAttr_info(attr_i, true_attr_filtered_idx, true_attr1_filtered, pred_attr1_filtered)
                                        counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)][str(attr_j)] = getAttr_info(attr_j, true_attr_filtered_idx, true_attr2_filtered, pred_attr2_filtered)
                                        counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)][str(attr_k)] = getAttr_info(attr_k, true_attr_filtered_idx, true_attr3_filtered, pred_attr3_filtered)
                                        counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)][str(attr_m)] = getAttr_info(attr_m, true_attr_filtered_idx, true_attr4_filtered, pred_attr4_filtered)
                                    

                                    if numberOfCombination > 4:
                                        for attr_n in range(attr_m +1 ,attrNum):

                                            true_attr5 = reshape_truelabel_vector[attr_n]
                                            pred_attr5 = reshape_predprob_vector[attr_n]
                                            # filter the data based on the true_attr1 where ==1 
                                            true_attr5_filtered_idx = numpy.where(true_attr5 ==1)
                                            pred_attr5_filtered_idx = numpy.where(pred_attr4 ==1)      
                                            # correctedPredictedImageIDs5 = {}              
                                            tmp =numpy.intersect1d(true_attr5_filtered_idx,pred_attr5_filtered_idx)
                                            correctedPredictedImageIDs5['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)  + '_' + str(attr_n)]  = numpy.intersect1d(tmp, correctedPredictedImageIDs4['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)] )
                                           
                                            # update the index list: true_attr1_filtered_idx
                                            true_attr_filtered_idx = numpy.intersect1d(true_attr_filtered_idx,true_attr5_filtered_idx)

                                            true_attr5_filtered = true_attr5[true_attr_filtered_idx]
                                            pred_attr5_filtered = pred_attr5[true_attr_filtered_idx]

                                            true_attr4_filtered = true_attr4[true_attr_filtered_idx]
                                            pred_attr4_filtered = pred_attr4[true_attr_filtered_idx]

                                            true_attr3_filtered = true_attr3[true_attr_filtered_idx]
                                            pred_attr3_filtered = pred_attr3[true_attr_filtered_idx]

                                            true_attr2_filtered = true_attr2[true_attr_filtered_idx]
                                            pred_attr2_filtered = pred_attr2[true_attr_filtered_idx]

                                            true_attr1_filtered = true_attr1[true_attr_filtered_idx]
                                            pred_attr1_filtered = pred_attr1[true_attr_filtered_idx]


                                            if numberOfCombination ==5  and len(true_attr_filtered_idx) > 0:
                                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)  + '_' + str(attr_n)] = {}
                                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)  + '_' + str(attr_n)]['imageIDs'] = numpy.array(true_attr_filtered_idx).tolist()
                                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)  + '_' + str(attr_n)]['imageIDs_correctPred'] = numpy.array(correctedPredictedImageIDs5['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)  + '_' + str(attr_n)] ).tolist()
                                                
                                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)  + '_' + str(attr_n)][str(attr_i)] = getAttr_info(attr_i,true_attr_filtered_idx,  true_attr1_filtered, pred_attr1_filtered)
                                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)  + '_' + str(attr_n)][str(attr_j)] = getAttr_info(attr_j,true_attr_filtered_idx,  true_attr2_filtered, pred_attr2_filtered)
                                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)  + '_' + str(attr_n)][str(attr_k)] = getAttr_info(attr_k,true_attr_filtered_idx,  true_attr3_filtered, pred_attr3_filtered)
                                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)  + '_' + str(attr_n)][str(attr_m)] = getAttr_info(attr_m,true_attr_filtered_idx,  true_attr4_filtered, pred_attr4_filtered)
                                                counter['attr_'+str(attr_i)+'_'+str(attr_j) + '_' + str(attr_k) + '_' + str(attr_m)  + '_' + str(attr_n)][str(attr_n)] = getAttr_info(attr_n,true_attr_filtered_idx,  true_attr5_filtered, pred_attr5_filtered)
                                            
                
                
        
        # print(counter)
        arrtibutes_combination[numberOfCombination] = counter
        numberOfCombination += 1

    
  

    #print("count_info done")
    return arrtibutes_combination

def getAttr_info(attr_id,true_attr_filtered_idx, true_attr1, pred_attr1):
    countInfo = {}
   
    # and case for TP        
    TP_res = numpy.logical_and(true_attr1,pred_attr1)*1

    countInfo['TP'] = int(numpy.sum(TP_res))
    
    # or case for TN
    TN_res = numpy.logical_not( numpy.logical_or(true_attr1,pred_attr1)*1 )*1
    countInfo['TN'] = int(numpy.sum(TN_res))

    # xor, xor  case for FP  , FN      
    res_tmp = numpy.logical_xor(true_attr1, pred_attr1)*1
    FP_res = numpy.logical_and(res_tmp, pred_attr1)*1
    FN_res = numpy.logical_and(res_tmp, true_attr1)*1

    countInfo['FP']= int(numpy.sum(FP_res))
    countInfo['FN'] = int(numpy.sum(FN_res))

    # other counting
    
    countInfo['truelabel_num']= int(numpy.sum(true_attr1))
    countInfo['predlabel_num']= int(numpy.sum(pred_attr1))    
    countInfo['attributeID'] = attr_id
    
    return countInfo


def query_get_tsne(params, conn, cursor):

    """Calculate tsne"""

    # Get all parameters
    n_components = params['parameters']['n_components'] # 2
    perplexity = params['parameters']['perplexity'] # 30
    early_exaggeration = params['parameters']['early_exaggeration'] # 12.0
    learning_rate = params['parameters']['learning_rate'] # 200.0
    n_iter = params['parameters']['n_iter'] # 1000
    metric = params['parameters']['metric'] # euclidean

    X = np.array(params['inputData'])

    X_embedded = TSNE(  n_components = n_components, perplexity = perplexity, early_exaggeration = early_exaggeration, learning_rate = learning_rate, n_iter = n_iter, metric = metric, init = 'pca').fit_transform(X);

    return X_embedded.tolist()


if __name__ == '__main__':

    
    # Read connection information from config.json
    with open('config.json') as config_file:
        conn_info = json.load(config_file)

    # Define database connection
    host = conn_info['host']
    dbname = conn_info['dbname']
    user = conn_info['user']
    port = conn_info['port']
    password = conn_info['password']


    # Create connection string
    conn_str =  "host='" + host + "' dbname='" + dbname + "' user='" + user + "' password='" + password + "' port='" + port + "'"

    # try connecting to postgresql database
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        print('Connected to postgresql database ...')
        print('')
    except:
        print('Please check the following! then run this code again: ')
        print('1. You have installed the PostgreSQL database.')
        print('2. The connection information in the config.json file are correct.')
        sys.exit()

    ## define the param and call function
    params = {"Data type": "synthetic", "Embedding method":"tsne"}
    
    # query_get_count_info(params, conn, cursor)
    query_get_mutual_info(params, conn, cursor)
    #print(res)


    print("done")  