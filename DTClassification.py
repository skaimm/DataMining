# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:01:02 2020

@author: uasmt
"""
from math import log
import operator

# karar agacı classımızı olusturalım
class DecisionTreeClassificaiton:
    def __init__(self,dataSet, labels):
        # modelimiz agac yapısını dictinary olarak tutacak
        self.model = self.createTreeModel(dataSet, labels)
    def createTreeModel(self,data,labels):
        # verileri ayıklayalım
        featureList = [feature[-1] for feature in data]
         # tüm sınıflar eşit olduğunda donguyu bitirelim
        if featureList.count(featureList[0]) == len(featureList):
            return featureList[0] 
        # verisetimiz'te başka özellik olmadığında donguyu bitirelim
        if len(data[0]) == 1:  
            return self.majorityCnt(featureList)
        
        # else kazanç ölcütüne devam edilir.
        
        #en iyi kazanca sahip niteligi alalım.
        bestFeature = self.findBestFeature(data)
        featureName = labels[bestFeature]

        #özyinemeli bir agac olusturalım.
        myTree = {featureName: {}}
        del (labels[bestFeature])
        featValues = [feature[bestFeature] for feature in data]
        #niteliklerin benzersiz degerğerlini bulalım ve özyenilemi olarak kazancları hesaplayamadevam edelim
        uniqueValues = set(featValues)
        for value in uniqueValues:
            copyLabels = labels[:]
            myTree[featureName][value] = self.createTreeModel(self.splitData(data, bestFeature, value), copyLabels)
        return myTree
    
    #entropi hesaplama niteliklerin sayılarını ve toplamını bulup entropi hesabı yapılır 
    def calcEntropy(self,data):
        lenOfData = len(data)
        labels = {}
        #nitelikteki labellerin sayısal degerlerini bul
        for feature in data:
            currentLabel = feature[-1]
            if currentLabel not in labels.keys(): labels[currentLabel] = 0
            labels[currentLabel] += 1
        entropy = 0.0
        # niteligin her label için entropi hesapla
        for key in labels:
            prob = float(labels[key]) / lenOfData
            entropy -= prob * log(prob, 2)
        return entropy
    
    #agac yapısınde derine dogru ilerken, kullandıgımız nitelikleri tekrar kullanmamak için ana verilerimizden ayrılır.
    def splitData(self,data, bestF, val):
        newData = []
        for feature in data:
            if feature[bestF] == val:
                newFeature = feature[:bestF]  
                newFeature.extend(feature[bestF + 1:])
                newData.append(newFeature)
        return newData
    
    def findBestFeature(self,data):
        lenOfData = len(data[0]) - 1
        #ilk hedef entropisini bulup,eniyikazan degerini ve onu veren niteligin indexini tanımlıyoruz.
        entropy = self.calcEntropy(data)
        bestGain = 0.0;
        bestFeature = -1
        # tüm nitelikleri tek tek dolasıyoruz
        for i in range(lenOfData):
            # niteligi liste olarak tutuyoruz.
            featureList = [feature[i] for feature in data]  
            # benzersiz degerliklerini alıyoruz
            uniqueValues = set(featureList)
            newEntropy = 0.0
            # ve her benzersiz degerlik için entropisini hesaplıyoruz
            for value in uniqueValues:
                newData = self.splitData(data, i, value)
                prob = len(newData) / float(len(data))
                newEntropy += prob * self.calcEntropy(newData)
            # kazancını hesaplıyoruz
            gain = entropy - newEntropy
            # eger kazanc onceki (ilk kazanc 0 olarak belirledik bestGain)buldugumuz kazanctan buyukse
            # bestGain i yeni niteligimizin index olarak guncelliyoruz
            if (gain > bestGain): 
                bestGain = gain
                bestFeature = i
        # ve en buyuk kazanlı niteligin index degerini donduruyoruz
        return bestFeature
    
    #baska nitelik kalmadıgında
    def majorityCnt(self,featureList):
        featureDict = {}
        for i in featureList:
            # son niteligimiz icindeki elemlarının
            # sayısal degerlrini bulalım
            if i not in featureDict.keys(): featureDict[i] = 0
            featureDict[i] += 1
        #sayısal degerlerine gore sıralat ve buyuk olanı dondur
        sortedfeatureDict = sorted(featureDict.items(), key=operator.itemgetter(1), reverse=True)
        return sortedfeatureDict[0][0]
    
    #modelimiz olustuktan sonra tahmin için
    def predict(self,model,label, testData):
        # olusturulan model agacında ki en ustteki bulunan nitelikten baslayarak
        for key,value in model.items():
            # tahmin içib verilen label ve verilerin esit olmasını saglar
            for index,name in enumerate(label):
                ans = testData[index]
                if name == key:
                    for ikey,ivalue in value.items():
                        # esitlenmeler yapıldıktan sonra bulunan degerimiz
                        if ikey == ans:
                            # eger hala bir dictionary ise agac yapısında derine inmeye devam ediyoruz
                            if isinstance(ivalue, dict):
                                self.predict(ivalue,label,testData)
                            # eger ki dictionary yapısı bitmisse sonuc bulunmustur.
                            else:
                                # ve onu print ediyoruz.
                                print('Prediction is ',ivalue)