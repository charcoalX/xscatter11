## this is python code for entropy, conditional entropy, and mutual information computation


import scipy
import numpy as np


def probability(x):
    '''input x is one dimenstional list, containing 0,1 binary element; 
    return is the marginal probablity of 1  in list x'''
    X = np.array(x)
    num_1 = np.sum(X)
    num_all = X.shape[0]
    return num_1 / num_all


def entropy(x):
    '''input x is a 1-dimenstional list, containing 0,1 binary element; 
    return is the entropy of list x'''
    px = probability(x)
    entr = - px * np.log2(px) - (1-px) * np.log2(1-px)
    return entr

def joint_prob(x,y):
    '''input x,y are two sequences (two 1-dimenstional lists), containing 0,1 binary element; 
    return is the joint probability P(x,y) list for four cases[P00,P01, P10,P11] '''
    X = np.array(x)
    Y = np.array(y)
    
    logical_or = np.logical_or(X,Y)*1
    # X0Y0
    not_logical_or = np.logical_not(logical_or)*1
    prob_00 = np.sum(not_logical_or) / not_logical_or.shape[0]

    logical_xor = np.logical_xor(X,Y)*1
    # X0Y1    
    andY_logical_xor = np.logical_and(logical_xor,Y)*1
    prob_01 = np.sum(andY_logical_xor) / andY_logical_xor.shape[0]

    # X1Y0
    andX_logical_xor = np.logical_and(logical_xor,X)*1
    prob_10 = np.sum(andX_logical_xor) / andX_logical_xor.shape[0]

    # X1Y1
    logical_and = np.logical_and(X,Y)*1
    prob_11 = np.sum(logical_and) / logical_and.shape[0]

    return prob_00, prob_01, prob_10, prob_11


def joint_entropy(x,y):
    '''input x,y are two sequences (two 1-dimenstional lists), containing 0,1 binary element; 
    return is the  joint Shannon entropy (in bits) of two discrete random variables X, Y (wiki)'''

    prob_00, prob_01, prob_10, prob_11 = joint_prob(x,y)
    joint_entro = - prob_00 *np.log2(prob_00) - prob_01 * np.log2(prob_01) - prob_10 * np.log2(prob_10) - prob_11 * np.log2(prob_11)

    return joint_entro


def conditional_entropy(x,y):
    '''input x,y are two sequences (two 1-dimenstional lists), containing 0,1 binary element; 
    return is the conditional entropy H(Y|X) 
    :: the amount of information needed to describe the outcome of a random variable,
     given that the value of another random variable X is knownn. (wiki)'''
    prob_x1 = probability(x)
    prob_x0 = 1 - prob_x1

    prob_00, prob_01, prob_10, prob_11 = joint_prob(x,y)
    conditional_entro = - prob_00 *np.log2((prob_00) / prob_x0)\
                        - prob_01 * np.log2((prob_01) /prob_x0 )\
                        - prob_10 * np.log2((prob_10) /prob_x1 )\
                        - prob_11 * np.log2((prob_11) /prob_x1)

    return conditional_entro

def mutual_information(x,y):
    '''input x,y are two sequences (two 1-dimenstional lists), containing 0,1 binary element; 
    return is the mutual infomation
    '''
    prob_x1 = probability(x)
    prob_x0 = 1 - prob_x1

    prob_y1 = probability(y)
    prob_y0 = 1 - prob_y1

    prob_00, prob_01, prob_10, prob_11 = joint_prob(x,y)

    mutual_info = prob_00 *np.log2((prob_00) / prob_x0 /prob_y0)\
                + prob_01 * np.log2((prob_01) /prob_x0 /prob_y1)\
                + prob_10 * np.log2((prob_10) /prob_x1 /prob_y0)\
                + prob_11 * np.log2((prob_11) /prob_x1 /prob_y1)

    return mutual_info



if __name__ == '__main__':

    X=[0,1,1,0,1,0,1,0,0,1,0,0]
    Y=[0,0,1,0,0,0,1,0,0,1,1,0]
    Z=[1,0,0,1,1,0,0,1,1,0,1,1]
    # print('*'*30)
    # print(X)
    # print(Y)
    # print(Z)

    # ## check entropy ####
    en_X = entropy(X)
    
    # en_X_scipy = scipy.stats.entropy([probability(X),(1-probability(X))],base=2)
    print("en_X =",en_X)
    
    # print("en_X_scipy = ",en_X_scipy)
    print("")

    # check joint prob 
    jt_prob = joint_prob(X,Y)
    print("jt_prob = ",jt_prob) 
    print("")


    ## check joint entropy
    jt_entro = joint_entropy(X,Y)
    print("jt_entro = ",jt_entro)
    print("")

    # check conditional entropy
    conditional_entr = conditional_entropy(X,Y)
 
    print("conditional_entr = ",conditional_entr)

    print("conditional_entr_math = ", jt_entro - en_X)
   

    print("")


    mutual_info = mutual_information(X,Y)

    print("mutual_info = ",mutual_info)

    print("mutual_info_math = ",en_X - conditional_entropy(Y,X))  # MI(x,y) = H(x) - H(x|y)



    print("done")
