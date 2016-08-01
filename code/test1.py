# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 15:46:41 2016

@author: Jennifer Ma
"""

import thinkstats2, thinkplot
import numpy as np
import nsfg
import math

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

means = []

def GetMeans(df):
    # get map from caseID to list of indices into the preg df
    map = nsfg.MakePregMap(df)
    
    for caseid in map:
        preg_indices = map[caseid]
        
        # ignore cases with less than 2 pregnancies
        if len(preg_indices) < 2:
            continue
        
        pregLngths = []
        for babyIndex in preg_indices:
            #print(df.prglngth[babyIndex])
            if df.outcome[babyIndex] == 1:
                #print('alive!')
                
                #if first baby, append preg duration to front of list, else add to end
                if df.birthord[babyIndex] == 1:
                    pregLngths.insert(0, df.prglngth[babyIndex])
                else:
                    pregLngths.append(df.prglngth[babyIndex])
     
        # ignore cases with less than 2 livebirths
        if len(pregLngths) < 2:
            continue
        
     
        # now use pregLngths to calculate duration diffs with first duration
        durDiffs = [] # duration differences
        for index in range(1, len(pregLngths)):
            durDiffs.append(pregLngths[0] - pregLngths[index])
        
        #print(durDiffs)
        mean = sum(durDiffs, 0.0) / len(durDiffs)
    #    print('mean: ', mean)
        
    #    firstLngths.
        means.append(math.floor(mean))


# plot a histogram of the means
hist = thinkstats2.Hist(means)
thinkplot.Hist(hist)
#thinkplot.Show(xlabel='value', ylabel='frequency')
thinkplot.Show(xlabel='mean diff', ylabel='frequency', axis=[-20,20, 0,1500])

print('mean', np.mean(means))


class HypoTest(thinkstats2.HypothesisTest):
    
    def __init__(self, data, df, preg_indices):    
        self.data = data
        self.df = df
        self.preg_indices = preg_indices
        
        self.MakeModel()
        self.actual = self.TestStatistic(data)
    
    def MakeModel(self):
        # make model for null hypothesis
        # idea: calc respondent's mean based on diff in durations with another randomly chosen baby, not necessarily the first baby
        # to simulate: get a random index 0 to (len(babies) and swap with first baby)
        #         
        
        # self.model = model
        
    
    def RunModel(self):
        # using the null hypothesis model data, 
        # want to simulate the hypothesis, which is that first babies are born a bit later than other babies from the same mother...
        # 
        # pick n respondents where n = size of data 
        # then calc diffs, calc their mean, add to means
        # return means
    
    def TestStatistic(self, data):
        return np.mean(data)
        
ht = HypoTest(means, df, preg_indices)
pval = ht.PValue()
    
    
# some notes:
#   70/30% train/test set...
    







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