# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 15:46:41 2016

@author: Jennifer Ma
"""

import thinkstats2, thinkplot
import numpy as np
import nsfg, nsfg2
import math
import time 
import copy
import random



df = nsfg.ReadFemPreg()
#print(df.columns)
df2 = nsfg2.ReadFemPreg()


#print(map[8186])
#print(df['prglngth'][8945])
#print(df['prglngth'][8946])

#caseid = 10229 # 1 livebirth, lots miscarriages
#caseid = 8186
#preg_indices = map[caseid]
#print(df.outcome[preg_indices].values)

# the test statistic here is the avg of the means of diffs for individual respondents

# runmodel should simulate the null hypothesis. here that means it should 

# our test data is df2
# we use df just for the actual test statistic
# we will use df2 for making a model of the null hypothesis, which is to calc differences with a random baby, not necessarily the first baby
# we dont really need make (prep the) model here, since the data is already together
# we just need runmodel (which simulates a random version of the N.H. model (for 1 test)), which will swap the first baby  at random in preg_indices, for each respondent,
# and then we run multiple tests (1000 iterations) of runmodel, getting the test statistic each time, and calculating p-value

# the p-value is the % of time that the N.H. model's T.S. reached the actual test statistic
# if < 0.01, S.S. by this hypothesis test. but need to consider false positive rate... which is the same as the p-value...
# its the p val is 1%, then there is a 1% chance that the experiment's result is a false positive?...

startTime = time.time()
#print(startTime)


class Dataset():
    
    def __init__(self, df):
        self.df = df
        self.map = self._makeMap()
        
    def _makeMap(self):
        # make map from caseID to list of indices into the preg df
        map = nsfg.MakePregMap(self.df)
        return self._cleanMap(map)
    
    def _cleanMap(self, map):
        # delete keys(respondents) with < 2 live births
#        map = self.map
        df = self.df
        
        mapCopy = copy.deepcopy(map)
        for caseid in map:
            preg_indices = map[caseid]
            
            # delete cases with less than 2 pregnancies
            if len(preg_indices) < 2:
                del mapCopy[caseid]                
                continue
            
            # now count live births and delete cases with < 2
            # ---------------            
            # count
            numLiveBirths = 0
            for babyIndex in preg_indices:
                #print(df.prglngth[babyIndex])
                if df.outcome[babyIndex] == 1: #live birth
                    numLiveBirths += 1
                    if numLiveBirths >= 2:
                        break            
            # delete cases with less than 2 livebirths
            if numLiveBirths < 2:
                del mapCopy[caseid]                
                continue
        
        return mapCopy
        
    
         
# make dataset classes
dataset  = Dataset(df)
dataset2 = Dataset(df2)

# so now i need to create and run the experiment to get the p-value for evaluating S.S
# then ill think about the error/power



class HypoTest():
    
    def __init__(self, dataset, dataset2):    
        self.dataset = dataset
        self.means = self.GetMeans(dataset)
#        print('len(self.means): ', len(self.means))
        #        self.MakeModel()
        self.actualTS = self.GetTestStat(self.means)
        
        
        self.dataset2 = dataset2
#        self.preg_indices = preg_indices
        

    
#    def MakeModel(self):
        # make model for null hypothesis
        # idea: calc respondent's mean based on diff in durations with another randomly chosen baby, not necessarily the first baby
        # to simulate: get a random index 0 to (len(babies) and swap with first baby)
        #         
        
        # self.model = model
        
        
    # the idea is...
    # using the null hypothesis model data, 
    # want to simulate the hypothesis, which is that first babies are born a bit later than other babies from the same mother...    
    def RunModel(self):
        
        # goal: for n where n=size of first dataset, swap first baby into a random position
        # return data in for TestStatistic to use         
        
        count = min(300, len(self.means))
#        count = len(self.means)
        # dataset2.df, map
        # go thru keys in map, swap
        means = []
        
        # shuffle dictionary (make list and shuffle)
        dsmap = dataset2.map
        keys = list(dsmap.keys())
        random.shuffle(keys)
        shuffledList = [(key, dsmap[key]) for key in keys]
        
        for caseid, preg_indices in shuffledList:
            
            count -= 1
            if count <= 0:
                break
        
            swappedPregLngths = self._GetPregLengthsAndSwap(preg_indices, self.dataset2)
            
            # calc mean
            mean = self._GetDiffsMean(swappedPregLngths)
            # add mean to means
            means.append(mean)
#            print('mean of means', mean)
        
        # get data and return it
        return means
        
    # ------
    # helper funcs    
        
    # only for dataset #1
        
    def GetMeans(self, dataset): 
        means = []
        for caseid in dataset.map:
        
            preg_indices = dataset.map[caseid]
            pregLngths = self._GetPregLengths(preg_indices, dataset)
            
            # calc mean
            mean = self._GetDiffsMean(pregLngths)
            # add mean to means
            means.append(mean)
        
        # get data and return it
        return means
    
    # only for dataset #1
    def _GetPregLengths(self, preg_indices, dataset):
        df = dataset.df     
        pregLngths = []
        for babyIndex in preg_indices:
            #print(df.prglngth[babyIndex])
            if df.outcome[babyIndex] == 1: # livebirth                                            
                if df.birthord[babyIndex] == 1: #if first baby,
                    # append preg duration to front of list                    
                    pregLngths.insert(0, df.prglngth[babyIndex])
                else: # else not first baby, just add to end
                    pregLngths.append(df.prglngth[babyIndex])
        return pregLngths
        
    
        
    def _GetPregLengthsAndSwap(self, preg_indices, dataset):
        df = dataset.df    
        pregLngths = []
        for babyIndex in preg_indices:
            #print(df.prglngth[babyIndex])
            if df.outcome[babyIndex] == 1: # livebirth                                            
                if df.birthord[babyIndex] == 1: #if first baby,
                    # append preg duration to front of list                    
                    pregLngths.insert(0, df.prglngth[babyIndex])
                else: # else not first baby, just add to end
                    pregLngths.append(df.prglngth[babyIndex])
        swappedPregLngths = self._Swap(pregLngths)
        return swappedPregLngths
                
    def _Swap(self, pregLngths):
        newPos = random.randint(0, len(pregLngths)-1)
        temp = pregLngths[newPos]
        pregLngths[newPos] = pregLngths[0]
        pregLngths[0] = temp
        return pregLngths
    
    def _GetDiffsMean(self, pregLngths):
        # calc 1st livebirth preg length - all others for same mother,
        # then get mean of that and return it
        # now use pregLngths to calculate duration diffs with first duration
        durDiffs = [] # duration differences
        for index in range(1, len(pregLngths)):
            durDiffs.append(pregLngths[0] - pregLngths[index])
        #print(durDiffs)
            
        mean = sum(durDiffs, 0.0) / len(durDiffs)
        return mean
        
    # ---------
    
    def GetTestStat(self, data):
#        print('actualTS - test stat:', np.mean(self.means) - np.mean(data))
        return np.std(data)
#        return np.mean(data)
        
    def GetPValue(self, iters=50):
        self.test_stats = [self.GetTestStat(self.RunModel()) 
                            for _ in range(iters)]
#        (print('test stat: ', x) for x in self.test_stats)
        count = sum(1 for x in self.test_stats if x >= self.actualTS)        
        return count / iters
        
        
ht = HypoTest(dataset, dataset2)

means = ht.means
meansBinned = [round(mean) for mean in means]

# plot a histogram of the means
hist = thinkstats2.Hist(meansBinned)
thinkplot.Hist(hist)
#thinkplot.Show(xlabel='value', ylabel='frequency')
thinkplot.Show(xlabel='Mean difference', ylabel='Frequency', axis=[-20,20, 0, 1500])
print('mean of means: ', np.mean(meansBinned))



time2 = time.time() - startTime
print('time taken 1: ', time2)


#pvalue = ht.GetPValue()
#print('pval: ', pvalue)


#time3 = time.time() - startTime - time2
#print('time taken for getpvalue: ', time3)


#def GetFalseNegRate(dataset, dataset2, num_runs=100):
#    
#    numFalseNegs = 0
#    
#    for _ in range(num_runs):
#        
#        
## resample with replacement the data from dataset and dataset2
#def Resample(data):
#    


#def MakeNormalProbPlot(values):
#    # plot the normal probability plot
#    valsNPArr = np.asarray(values)
#    mean = valsNPArr.mean()
#    std = valsNPArr.std()
#    #mean = sample.mean()
#    #std = sample.std()
#    xs = [-4, 4] # range for x-axis: # of std devs away from mean
#    fxs, fys = thinkstats2.FitLine(xs, inter=mean, slope=std)
#    # plot normal model/line of fit based on sample's mean and std. if sample fits this it is most likely a normal dist.
#    thinkplot.Plot(fxs, fys, color='gray', label='model')
#    # plot sample compared to line of fit
#    # get sorted sample and random set of std values of same count
#    stdDraws, sortedVals = thinkstats2.NormalProbability(values)
#    #stdDraws, sortedSample = thinkstats2.NormalProbability(sample)    
#    thinkplot.Plot(stdDraws, sortedVals)


## get a random sample from difference means
#sample = np.random.choice(means, 1000, replace=True)

#thinkplot.PrePlot(2, cols=2)
#MakeNormalProbPlot(means)
#thinkplot.SubPlot(2)
### plot log of values to see if log is a better dist fit
#MakeNormalProbPlot(np.log(means))


# normal dist seems a better fit...
#wait, it should be normal.. since its based on means, and bc of the CLT


# further confirmation normal is a pretty good fit:
#def PlotNormalPDFAndKDE(values):
#
#    valsNPArr = np.asarray(values)
#    mean = valsNPArr.mean()
#    std = valsNPArr.std()
#    
#    
#    pdf = thinkstats2.NormalPdf(mean, std)
#    thinkplot.Pdf(pdf, label='normal')
#        
#    sample = np.random.normal(mean, std, 500)
#    pdf = thinkstats2.EstimatedPdf(sample)
#    thinkplot.Pdf(pdf)
#    
#PlotNormalPDFAndKDE(means)



#pmf = thinkstats2.Pmf(means)
#thinkplot.Pmf(pmf)
#thinkplot.Hist(pmf) #plots pmf as bar graph

#for diff, freq in hist.Items():
#    print(diff, freq)

#for diff in sorted(hist.Values()):
#    print(diff, hist.Freq(diff))