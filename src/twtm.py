'''
Created on Jul 19, 2016

@author: shuangyinli
'''
from numpy import *
import numpy as np
import random
import time
import multiprocessing
from multiprocessing import Process,Manager
import copy
import sys
import math


def log_sum(log_a, log_b):
    if log_a < log_b:
        return log_b + log(1+exp(log_a-log_b))
    else:
        return log_a + log(1+exp(log_b-log_a))
def trigamma(x):
    x = x + 6
    p = 1 / (x * x)
    p = (((((0.075757575757576 * p - 0.033333333333333) * p + 0.0238095238095238) * p - 0.033333333333333) * p + 0.166666666666667) * p + 1) / x + 0.5 * p
    for i in range(6):
        x = x - 1
        p = 1 / (x * x) + p
    return p
def log_gamma(x):
    PI = math.acos(-1.0)
    tmp = (x - 0.5) * log(x + 4.5) - (x + 4.5)
    ser = 1.0 + 76.18009173 / (x + 0) - 86.50532033 / (x + 1) + 24.01409822 / (x + 2) - 1.231739516 / (x + 3) + 0.00120858003 / (x + 4) - 0.00000536382 / (x + 5)
    return tmp + log(ser * sqrt(2 * PI))
def digamma(x):
    r = 0.0
    while x <=5:
        r -= 1/x
        x += 1
    f = 1.0 / (x * x)
    t = f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0 + f * (-1.0 / 132.0 + f * (691.0 / 32760.0 + f * (-1.0 / 12.0 + f * 3617.0 / 8160.0)))))))
    return r + log(x) - 0.5 / x + t
def norm2(vec1, vec2, dim):
    ret =0
    for i in range(dim):
        ret += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i])
    return ret
    
class Document():
    def __init__(self, num_tags_, num_words_, num_topics_, lik_,
                tags_ptr_, words_ptr_, words_cnt_ptr_):
        '''
        Constructor
        '''
        self.num_tags = num_tags_ # tag num in each doc
        self.num_words = num_words_
        self.num_topics = num_topics_
        self.lik = lik_
        self.tags_ptr = tags_ptr_
        self.words_ptr = words_ptr_
        self.words_cnt_ptr = words_cnt_ptr_
        self.xi = [0 for i in range(self.num_tags)]
        self.log_gamma = np.ndarray(shape = (self.num_words, self.num_topics), dtype = float)
        self.topic = [0 for i in range(self.num_topics)]
        self.Document_init()

    def Document_init(self):
        for i in range(self.num_tags):
            self.xi[i] = random.random()+0.5
        for i in range(self.num_words):
            for k in range(self.num_topics):
                self.log_gamma[i][k] = log(1.0 / self.num_topics)
        
class Model():
    def __init__(self, model_root_, num_docs_,num_words_,num_topics_,num_tags_,num_all_words_):
        '''
        Constructor
        '''
        self.num_docs = num_docs_
        self.num_words=num_words_
        self.num_topics=num_topics_
        self.num_tags=num_tags_ # all the tags number
        self.num_all_words=num_all_words_
        self.pi = [0.0 for i in range(self.num_tags)]
        self.log_theta=  np.ndarray(shape=(self.num_tags,self.num_topics), dtype=float )
        self.log_phi= np.ndarray(shape=(self.num_topics,self.num_words), dtype=float )
        self.model_root = model_root_
        self.model_init()
        
    def model_init(self):
        for i in range(self.num_tags):
            self.pi[i] = random.random() * 0.5 +1
            temp = 0.0
            for k in range(self.num_topics):
                v = random.random()
                temp += v
                self.log_theta[i][k] = v
            for k in range(self.num_topics):
                self.log_theta[i][k] = log(self.log_theta[i][k] / temp)
        for k in range(self.num_topics):
            for w in range(self.num_words):
                self.log_phi[k][w] = log(1.0 / self.num_words)
    
    def read_model_info(self):
        pass
    
    def load_mat(self):
        pass
    
    def save_model(self, num_round):
        if num_round != -1:
            pi_file = self.model_root+"pi."+str(num_round)
            theta_file = self.model_root+"theta."+str(num_round)
            phi_file = self.model_root+"phi."+str(num_round)
        else:
            pi_file = self.model_root+"pi.final"
            theta_file = self.model_root+"theta.final"
            phi_file = self.model_root+"phi.final"
        self.print_mat(self.log_phi, self.num_topics, self.num_words, phi_file)
        self.print_mat(self.log_theta, self.num_tags, self.num_topics, theta_file)
        self.print_array(self.pi, self.num_tags, pi_file)
    
    def print_mat(self, mat, row, col, filename):
        outputfile = open(filename, "w", encoding = "utf-8")
        for i in range(row):
            for j in range(col):
                outputfile.write(str(mat[i][j]) + " ")
            outputfile.write("\n")
        outputfile.flush()
        outputfile.close()
        
    def print_array(self, mat, row, filename):
        outputfile = open(filename, "w", encoding = "utf-8")
        for i in range(row):
            outputfile.write(str(mat[i]) + " ")
        outputfile.flush()
        outputfile.close()
    
    def print_model_info(self):
        outputfile = open(self.model_root+"model.info", "w", encoding="utf-8")
        outputfile.write("num_tags: " + str(self.num_tags) + "\n")
        outputfile.write("num_words: " + str(self.num_words) + "\n")
        outputfile.write("num_topics: " + str(self.num_topics) + "\n")
        outputfile.close()

class Configuration():
    def __init__(self, settingsfile):
        '''
        Constructor
        '''
        self.pi_learn_rate = 0.00001
        self.max_pi_iter=100
        self.pi_min_eps=1e-5
        self.max_xi_iter=100
        self.xi_min_eps=1e-5
        self.xi_learn_rate = 10
        self.max_em_iter=30
        self.num_threads=1
        self.max_var_iter=30
        self.var_converence = 1e-6
        self.em_converence = 1e-4
        self.read_settingfile(settingsfile)
        
    def read_settingfile(self,settingsfile):
        settingslist = open(settingsfile, "r", encoding = "utf-8")
        for line in settingslist:
            confname = line.split()[0]
            confvalue = line.split()[1]
            if confname == "pi_learn_rate":
                self.pi_learn_rate = float(confvalue)
            if confname == "max_pi_iter":
                self.max_pi_iter = int(confvalue)
            if confname == "pi_min_eps":
                self.pi_min_eps = float(confvalue)
            if confname == "max_xi_iter":
                self.max_xi_iter = int(confvalue)
            if confname == "xi_learn_rate":
                self.xi_learn_rate = float(confvalue)
            if confname == "xi_min_eps":
                self.xi_min_eps = float(confvalue)
            if confname == "max_em_iter":
                self.max_em_iter = int(confvalue)
            if confname == "num_threads":
                self.num_threads = int(confvalue)
            if confname == "var_converence":
                self.var_converence = float(confvalue)
            if confname == "max_var_iter":
                self.max_var_iter = int(confvalue)
            if confname == "em_converence":
                self.em_converence = float(confvalue)
             
class TWTM():
    def __init__(self,filename, num_topics, settingsfile, model_root_):
        '''
        Constructor
        '''
        self.corpus = []
        self.num_docs=0
        self.num_tags=0 # all the tag number in the corpus
        self.num_words=0
        self.num_all_words=0
        self.num_topics = num_topics
        self.model_root = model_root_
        
        self.read_data(filename)
        self.model = Model(self.model_root, self.num_docs, self.num_words, self.num_topics,self.num_tags, self.num_all_words)

        self.configuration = Configuration(settingsfile)
        self.begin_twtm()
        
    def read_data(self, filename):
        datalist = open(filename, "r", encoding = "utf-8").readlines()
        for onedata in datalist:
            labelslist = onedata.split("@")[0]
            wordslist = onedata.split("@")[1]
            doc_num_tags = int(labelslist.split()[0])
            tags_ptr_ = [int(m) for m in labelslist.split()[1:]] 
            num_word = int(wordslist.split()[0])
            words = wordslist.split()[1:]
            tags_ptr = []
            words_ptr = []
            words_cnt_ptr = []
            for t in range(doc_num_tags):
                tags_ptr.append(int(tags_ptr_[t]))
            if self.num_tags < max(tags_ptr):
                self.num_tags = max(tags_ptr)
            for i in range(num_word):
                id_count = words[i].split(":")
                words_ptr.append(int(id_count[0]))
                words_cnt_ptr.append(int(id_count[1]))
                self.num_all_words += words_cnt_ptr[i]
            if self.num_words < max(words_ptr):
                self.num_words = max(words_ptr)
            doc = Document(doc_num_tags, num_word, self.num_topics, 100, tags_ptr_, words_ptr,words_cnt_ptr)
            self.corpus.append(doc)
            self.num_docs += 1
        self.num_tags += 1
        self.num_words +=1
        if self.corpus[1].num_tags != len(self.corpus[1].tags_ptr):
            print("the number of tags in a documet doesn't equal with its ptr..")
            exit(0)
        if self.corpus[1].num_words != len(self.corpus[1].words_ptr):
            print("the number of words in a documet doesn't equal with its ptr..")
        #     
        print("num_docs: "+str(self.num_docs)+", num_tags: "+str(self.num_tags)+", num_words: "+str(self.num_words)+" \n")
    
    def save_parameters_docs(self, num_round):
        if num_round != -1:
            xi_file = self.model.model_root+"xi."+str(num_round)
            topic_dis_file = self.model.model_root+"topic_dis_docs."+str(num_round)
        else:
            xi_file = self.model.model_root+"xi.final"
            topic_dis_file = self.model.model_root+"topic_dis_docs.final"
        xi_fp = open(xi_file, "w", encoding = "utf-8")
        topic_dis_fp = open(topic_dis_file,"w", encoding = "utf-8")
        
        for doc in self.corpus:
            for i in range(doc.num_tags):
                xi_fp.write(str(doc.tags_ptr[i]) +":"+ str(doc.xi[i]))
            xi_fp.write("\n")
            
            for k in range(self.num_topics):
                topic_dis_fp.write(str(doc.topic[k]) + ' ')
            topic_dis_fp.write("\n")
            
        xi_fp.close()
        topic_dis_fp.close()
    
    def likelihood(self):
        lik = 0.0 
        for d in range(self.num_docs):
            temp_lik, return_doc = self.compute_doc_likelihood(self.corpus[d])
            lik += temp_lik
            #self.corpus[d].lik = temp_lik
        return lik
    
    def splitlikelihood(self, dataSplit, likreturn_dataSplit, likreturn_dataSplit_likvalue):
        splitlen = len(dataSplit)
        for d in range(splitlen):
            temp_lik, return_doc = self.compute_doc_likelihood(dataSplit[d])
            likreturn_dataSplit_likvalue.value += temp_lik
            likreturn_dataSplit.append(return_doc)
    
    def run_multiprocesses_likelihood(self):
        lik = 0.0
        workers = []
        workers_no = self.configuration.num_threads
        corpusSplitlist = self.split_average_data(workers_no)
        
        likmanager = Manager()
        ManagerReturn_corpusSplitlist = []
        ManagerReturn_corpusSplitlist_lik = []
        for dataSplit in corpusSplitlist:
            likreturn_dataSplit = likmanager.list()
            likreturn_dataSplit_likvalue = likmanager.Value("",0.0)
            worker = Process(target=self.splitlikelihood, args=(dataSplit, likreturn_dataSplit, likreturn_dataSplit_likvalue))
            worker.start()
            workers.append(worker)
            ManagerReturn_corpusSplitlist.append(likreturn_dataSplit)
            ManagerReturn_corpusSplitlist_lik.append(likreturn_dataSplit_likvalue)
        for w in workers:
            w.join()
        
        # compute all the likelihood for the splits:
        for v in ManagerReturn_corpusSplitlist_lik:
            lik += v.value
        # update all the docs into corpus, since we compute the doc distribution in likelihood()
        self.corpus.clear()
        for dataSplit in ManagerReturn_corpusSplitlist:
            for doc in dataSplit:
                self.corpus.append(doc)
        
        return lik
    
    def get_pi_function(self):
        pi_function_value = 0.0
        num_docs = self.model.num_docs
        pi = self.model.pi
        for d in range(num_docs):
            sigma_pi = 0.0
            sigma_xi = 0.0
            doc = self.corpus[d]
            for i in range(doc.num_tags):
                sigma_pi += pi[doc.tags_ptr[i]]
                sigma_xi += doc.xi[i]
            pi_function_value += log_gamma(sigma_pi)
            for i in range(doc.num_tags):
                tagid = doc.tags_ptr[i]
                pi_function_value -= log_gamma(pi[tagid])
                pi_function_value += (pi[tagid] -1) * (digamma(doc.xi[i]) - digamma(sigma_xi) )
        return pi_function_value
    
    def get_descent_pi(self):
        num_tags = self.model.num_tags
        num_docs = self.model.num_docs
        descent_pi = [0.0 for i in range(num_tags)]
        pi = self.model.pi
        for d in range(num_docs):
            sigma_pi = 0.0
            doc = self.corpus[d]
            doc_num_tags = doc.num_tags
            sigma_xi = 0.0
            for i in range(doc_num_tags):
                sigma_pi += pi[doc.tags_ptr[i]]
                sigma_xi += doc.xi[i]
            for i in range(doc_num_tags):
                tagid = doc.tags_ptr[i]
                pis = pi[tagid]
                descent_pi[tagid] += digamma(sigma_pi) - digamma(pis) + digamma(doc.xi[i]) - digamma(sigma_xi)
        return descent_pi
    
    def learn_pi(self):
        num_round = 0
        num_tags = self.model.num_tags
        last_pi = [0.0 for i in range(num_tags)]
        descent_pi = [0.0 for i in range(num_tags)]
        #
        z = -1
        num_wait_for_z = 0
        while z < 0 and num_wait_for_z <=20:
            for i in range(num_tags):
                self.model.pi[i] = random.random() *2
            z = self.get_pi_function()
            #print("wait for z >=0 \n")
            num_wait_for_z += 1
        #    
        last_z = 0
        learn_rate = self.configuration.pi_learn_rate
        eps =10000
        max_pi_iter = self.configuration.max_pi_iter
        pi_min_eps = self.configuration.pi_min_eps
        has_neg_value_flag = False
        while num_round < max_pi_iter and eps > pi_min_eps:
            last_z = z
            last_pi = copy.deepcopy(self.model.pi)
            descent_pi = self.get_descent_pi()
            for i in range(num_tags):
                self.model.pi[i] += learn_rate * descent_pi[i]
                if self.model.pi[i] <0:
                    has_neg_value_flag = True
            z = self.get_pi_function()
            if has_neg_value_flag or last_z > z:
                learn_rate *= 0.5
                z = last_z
                self.model.pi = copy.deepcopy(last_pi)
                eps = 1000.0
            else:
                eps = norm2(last_pi, self.model.pi, num_tags)
            num_round +=1
    
    def learn_theta_phi(self):
        num_docs = self.model.num_docs
        num_topics = self.model.num_topics
        num_words = self.model.num_words
        num_tags = self.model.num_tags
        reset_theta_flag = np.full(shape = (num_tags, num_topics),  dtype = bool, fill_value = False)
        reset_phi_flag = np.full(shape = (num_topics, num_words),  dtype = bool, fill_value = False)
        for d in range(num_docs):
            doc = self.corpus[d]
            doc_num_tags = doc.num_tags
            doc_num_words = doc.num_words
            sigma_xi = 0
            for i in range(doc_num_tags):
                sigma_xi += doc.xi[i]
            for i in range(doc_num_tags):
                tagid = doc.tags_ptr[i]
                for k in range(num_topics):
                    for j in range(doc_num_words):
                        if reset_theta_flag[tagid][k] is False:
                            reset_theta_flag[tagid][k] = True
                            self.model.log_theta[tagid][k] = log(doc.words_cnt_ptr[j]) + doc.log_gamma[j][k] + log(doc.xi[i]) - log(sigma_xi)
                        else:
                            self.model.log_theta[tagid][k] = log_sum(self.model.log_theta[tagid][k], log(doc.words_cnt_ptr[j]) + doc.log_gamma[j][k] + log(doc.xi[i]) - log(sigma_xi) )
            for k in range(num_topics):
                for i in range(doc_num_words):
                    wordid = doc.words_ptr[i]
                    if reset_phi_flag[k][wordid] is False:
                        reset_phi_flag[k][wordid] = True
                        self.model.log_phi[k][wordid] = log(doc.words_cnt_ptr[i]) + doc.log_gamma[i][k]
                    else:
                        self.model.log_phi[k][wordid] = log_sum(self.model.log_phi[k][wordid], log(doc.words_cnt_ptr[i]) + doc.log_gamma[i][k])
        
        #
        self.normalize_log_matrix_rows(self.model.log_theta, num_tags, num_topics)
        self.normalize_log_matrix_rows(self.model.log_phi, num_topics, num_words)
        pass
    
    def normalize_log_matrix_rows(self, log_mat, rows, cols):
        for i in range(rows):
            temp = log_mat[i][0]
            for j in range(cols-1):
                temp = log_sum(temp, log_mat[i][j+1])
            for j in range(cols):
                log_mat[i][j] -= temp
    
    def get_xi_function(self, doc):
        xi_function_value = 0.0
        num_tags = doc.num_tags
        sigma_xi = 0
        pi = self.model.pi
        log_theta = self.model.log_theta
        for i in range(num_tags):
            sigma_xi += doc.xi[i]
        for i in range(num_tags):
            xi_function_value += (pi[doc.tags_ptr[i]] - doc.xi[i]) * (digamma(doc.xi[i]) - digamma(sigma_xi)) + log_gamma(doc.xi[i])
        xi_function_value -= log_gamma(sigma_xi)
        doc_num_words = doc.num_words
        num_topics = self.model.num_topics
        sum_log_theta = [0 for i in range(num_topics)]
        for k in range(num_topics):
            temp = 0
            for j in range(num_tags):
                temp += log_theta[doc.tags_ptr[j]][k] * doc.xi[j] / sigma_xi
                sum_log_theta[k] = temp
        for i in range(doc_num_words):
            for k in range(num_topics):
                xi_function_value +=  sum_log_theta[k] * exp(doc.log_gamma[i][k]) * doc.words_cnt_ptr[i]
        return xi_function_value
    
    def get_descent_xi(self, doc):
        sigma_xi = 0.0
        sigma_pi = 0.0
        num_tags = doc.num_tags
        descent_xi = [0 for i in range(num_tags)]
        for i in range(num_tags):
            sigma_xi += doc.xi[i]
            sigma_pi += self.model.pi[doc.tags_ptr[i]]
        
        for i in range(num_tags):
            descent_xi[i] = trigamma(doc.xi[i]) * ( self.model.pi[doc.tags_ptr[i]] - doc.xi[i])
            descent_xi[i] -= trigamma(sigma_xi) * (sigma_pi - sigma_xi)
        
        doc_num_words = doc.num_words
        num_topic = self.num_topics
        log_theta =self.model.log_theta
        sum_log_theta = [0.0 for i in range(num_topic)]
        for k in range(num_topic):
            for i in range(num_tags):
                tag_id = doc.tags_ptr[i]
                sum_log_theta[k] += log_theta[tag_id][k] * doc.xi[i]
       
        sum_gamma_array = [0.0 for i in range(num_topic)]
        for k in range(num_topic):
            sum_gamma_array[k] = 0
            for i in range(doc_num_words):
                sum_gamma_array[k] += exp(doc.log_gamma[i][k]) * doc.words_cnt_ptr[i]
        
        for j in range(num_tags):
            for k in range(num_topic):
                temp = 0
                temp += log_theta[doc.tags_ptr[j]][k] * sigma_xi
                temp -= sum_log_theta[k]
                temp = sum_gamma_array[k] * (temp/(sigma_xi * sigma_xi))
                descent_xi[j] += temp
        
        return descent_xi
        
    
    def inference_xi(self, doc):
        num_tags = doc.num_tags
        # init_xi
        for i in range(num_tags):
            doc.xi[i] = random.random()
        z = 0
        learn_rate = self.configuration.xi_learn_rate
        eps = 10000
        num_round =0
        max_xi_iter = self.configuration.max_xi_iter
        xi_min_eps = self.configuration.xi_min_eps
        last_z = 0
        last_xi = []
        while num_round < max_xi_iter and eps > xi_min_eps:
            z = self.get_xi_function(doc)
            last_z = z
            last_xi = copy.deepcopy(doc.xi)
            descent_xi =self.get_descent_xi(doc)
            has_neg_value_flag = False
            for i in range(num_tags):
                doc.xi[i] += learn_rate * descent_xi[i]
                if doc.xi[i] < 0:
                    has_neg_value_flag = True
           
            if has_neg_value_flag is True or last_z > self.get_xi_function(doc):
                learn_rate *= 0.2
                z = last_z
                eps = 10000
                doc.xi = copy.deepcopy(last_xi)
            else:
                eps = norm2(last_xi, doc.xi, num_tags)
            num_round += 1
    
    def inference_gamma(self, doc):
        log_theta = self.model.log_theta
        log_phi = self.model.log_phi
        num_tags = doc.num_tags
        num_topics = self.num_topics
        doc_num_words = doc.num_words
        log_gamma = doc.log_gamma
        theta_xi = [0.0 for k in range(num_topics)]
        sigma_xi = 0
        for i in range(num_tags):
            sigma_xi += doc.xi[i]
        
        for k in range(num_topics):
            temp = 0.0
            for i in range(num_tags):
                temp+= doc.xi[i] / sigma_xi * log_theta[doc.tags_ptr[i]][k]
            theta_xi[k] = temp
        
        for i in range(doc_num_words):
            wordid = doc.words_ptr[i]
            sum_log_gamma = 0
            for k in range(num_topics):
                temp = log_phi[k][wordid] + theta_xi[k]
                log_gamma[i][k] = temp
                if k ==0:
                    sum_log_gamma = temp
                else:
                    sum_log_gamma = log_sum(sum_log_gamma, temp)
            for k in range(num_topics):
                log_gamma[i][k] -= sum_log_gamma
        pass
    
    def compute_doc_likelihood(self, doc):
        log_topic = doc.topic
        log_theta = self.model.log_theta
        log_phi = self.model.log_phi
        num_topics = self.num_topics
        reset_log_topic = [False for i in range(num_topics)]
        for k in range(num_topics):
            log_topic[k] = 0
        sigma_xi = 0
        xi = doc.xi
        doc_num_tags = doc.num_tags
        lik = 0.0 
        for i in range(doc_num_tags):
            sigma_xi += xi[i]
        
        for i in range(doc_num_tags):
            tagid = doc.tags_ptr[i]
            for k in range(num_topics):
                if reset_log_topic[k] is False:
                    log_topic[k] = log_theta[tagid][k] + log(xi[i]) - log(sigma_xi)
                    reset_log_topic[k] = True
                else:
                    log_topic[k] = log_sum(log_topic[k],  log_theta[tagid][k] + log(xi[i]) - log(sigma_xi))
        
        doc_num_words = doc.num_words
        for i in range(doc_num_words):
            temp = 0
            wordid = doc.words_ptr[i]
            temp = log_topic[0] + log_phi[0][wordid]
            for k in range(num_topics-1):
                temp = log_sum(temp, log_topic[k+1] + log_phi[k+1][wordid])
                # because 0 is already added, so from k+1
            lik += temp * doc.words_cnt_ptr[i]
        
        doc.lik = lik
        return lik, doc
    
    def inference(self, doc):
        var_iter =0
        lik_old = -10000000
        converged = 1
        lik = 0
        while (converged > self.configuration.var_converence) and var_iter < self.configuration.max_var_iter:
            var_iter += 1
            self.inference_xi(doc)
            self.inference_gamma(doc)
            lik, return_doc = self.compute_doc_likelihood(doc)
            converged = (lik_old - lik) / lik_old
            lik_old = lik
        return doc

    def inferenceDatasplit(self, datasplit, managerDoclist):
        datasize = len(datasplit)
        for i in range(datasize):
            doc = self.inference(datasplit[i])
            managerDoclist.append(doc)
        
    def split_average_data(self, thread_no):
        fn = len(self.corpus)//thread_no
        rn = len(self.corpus)%thread_no
        ar = [fn+1]*rn+ [fn]*(thread_no-rn)
        si = [i*(fn+1) if i<rn else (rn*(fn+1)+(i-rn)*fn) for i in range(thread_no)]
        corpusSplitlist = [self.corpus[si[i]:si[i]+ar[i]] for i in range(thread_no)]
        return corpusSplitlist
    
    def run_multiprocesses_inference(self):
        workers = []
        workers_no = self.configuration.num_threads
        corpusSplitlist = self.split_average_data(workers_no)
        manager = Manager()
        ManagerReturn_corpusSplitlist = []
        for dataSplit in corpusSplitlist:
            return_dataSplit = manager.list()
            worker = Process(target=self.inferenceDatasplit, args=(dataSplit, return_dataSplit))
            worker.start()
            workers.append(worker)
            ManagerReturn_corpusSplitlist.append(return_dataSplit)
        for w in workers:
            w.join()
        
        self.corpus.clear()
        # after all the processes, update the corpus using the ManagerReturn_corpusSplitlist
        for dataSplit in ManagerReturn_corpusSplitlist:
            for doc in dataSplit:
                self.corpus.append(doc)
    
    def begin_twtm(self):
        self.model.print_model_info()
        learn_begin_time = time.time()
        num_round = 0
        print("compute the likelihood...")
        #lik1 = self.likelihood() 
        lik = self.run_multiprocesses_likelihood() 

        plik = 0.0
        likehood_record = []
        likehood_record.append(lik)
        converged = 1
        while num_round < self.configuration.max_em_iter and (converged < 0 or converged > self.configuration.em_converence):
            cur_round_begin_time = time.time()
            plik = lik
            print("Round %d begin... "%num_round)
            print("inference...")
            self.run_multiprocesses_inference()
            print("learn pi .... ")
            self.learn_pi()
            print("learn theta .... ")
            self.learn_theta_phi()
            print("compute the likelihood...")
            lik = self.run_multiprocesses_likelihood()
            perplexity = exp(-lik/self.model.num_all_words)
            converged = (plik - lik) / plik
            if converged < 0:
                self.configuration.max_var_iter *=2
            cur_round_cost_time = time.time() - cur_round_begin_time
            print("Round "+str(num_round)+" : likehood= "+str(lik)+" . last_likehood= "+str(plik)+" . perplexity= "+str(perplexity)+" converged= "+str(converged)+" . cost_time= "+str(cur_round_cost_time)+" secs.\n")
            num_round += 1
            likehood_record.append(lik)
            if num_round % 10 ==0:
                self.model.save_model(num_round)
                self.save_parameters_docs(num_round)
            
        learn_cost_time = time.time() - learn_begin_time
        print("All the round learning is over, and cost %f seconds."%learn_cost_time)
        self.model.save_model(-1)
        self.save_parameters_docs(-1)
    
    def infer_twtm(self):
        pass
        
if __name__ == '__main__':
    if (len(sys.argv) != 6):
        print("usage1: twtm.py est <input data file> <setting.txt> <num_topics> <model save dir>\n")
        exit(0)
    inputfile = sys.argv[2]
    settingsfile = sys.argv[3]
    num_topics = int(sys.argv[4])
    model_root = sys.argv[5]
    if model_root.endswith("/") is False:
        model_root = model_root+"/"
    
    TWTM(inputfile, num_topics, settingsfile, model_root)
