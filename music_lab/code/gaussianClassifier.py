from os import listdir   
from numpy import array, mean, cov, append, linalg, tile, dot, transpose, zeros, argmin
from random import shuffle

# classifies music my genre 
# returns confusion matrix and classification accuracy for randomly partitioned data
def processMusic(genres, data):

    # Split data into training and test sets
    dSize = len(data[0])
    rIdx = range(dSize)
    mix = shuffle(rIdx)
    #lets create a random samble for both the training and then the remainder becomes the test
    trainIdx = rIdx[0:int(dSize*0.8)]
    testIdx  = rIdx[int(dSize*0.8)+1:]
    
    #Step 2 - Learning Gaussian Model for each Genre
    gModel = list()
    for g in range(len(genres)):
      trainMat = data[g][trainIdx[0]]['featureMat']
      for i in range(1,len(trainIdx)):
        trainMat = append(trainMat, data[g][trainIdx[i]]['featureMat'], axis =0)
      gModel.append({'mean':mean(trainMat,0), 'cov':cov(trainMat,rowvar=0),'icov':linalg.inv(cov(trainMat,rowvar=0))})

    
    #Step 3: Calculating Average Unnormalized Likelihood for each test song and genre model
    meanUNLL = zeros((len(genres),len(testIdx),len(genres)))
    guess = zeros((len(genres),len(testIdx)))
    for gs in range(len(genres)):
      for t in range(len(testIdx)):
        ts = testIdx[t]
        x = data[gs][ts]['featureMat']
        [r, c] =x.shape        
        for m in range(len(genres)):
          unll = zeros((r,1))
          
          for v in range(r):
            diff = (x[v] - gModel[m]['mean'])
            res = dot(diff, gModel[m]['icov'])
            res = dot(res, transpose(diff))
            unll[v] = res
          

          meanUNLL[gs][t][m] = mean(unll)
        guess[gs][t] = argmin(meanUNLL[gs][t])  
    
    #Step 4 Evaluate Results
    [cfm, acc] = createConfusionMatrix(guess)
    print "Trial Accuracy = ", acc
    return cfm, acc     
    
    
    
# loads metadata (song & artist name) and audio feature vectors for all songs
# format:
#    # Count On Me - Bruno Mars
#    0.0,171.13,9.469,-28.48,57.491,-50.067,14.833,5.359,-27.228,0.973,-10.64,-7.228
#    29.775,-90.263,-47.86,14.023,13.797,189.87,50.924,-31.823,-45.63,104.501,82.114,-13.67
#
# returns data in for of "list of lists of dict" where 
#   first index is the genre and the second index corresponds to a song
#   dict contains 'song', 'artist' , and 'featureMat'
    
def loadData(dataDir, genres):    
    data = list()
    for g in range(len(genres)):
      genreDir = dataDir+"/"+genres[g]
      data.append(list())
      sFiles = listdir(genreDir)
      for s in range(len(sFiles)):
        
        sFile = genreDir+"/"+sFiles[s]
        f = open(sFile)
        lines = f.readlines()
        meta = lines[0].replace("#","").split("-")
        songDict = {}
        
        #read in matrix of values starting from second line in data file
        mat = list()
        for i in range(1,len(lines)):
          vec = lines[i].split(",")
          for j in range(len(vec)):
            vec[j] = float(vec[j])
          mat.append(vec)  
            
          
        songDict['featureMat'] = array(mat)  
        data[g].append(songDict)
    
    return data 
      

# returns predicted-x-expected confusion matrix and classification accuracy
#  -assumes that the row index is the correct label
def createConfusionMatrix(resultMat):
    [rows, cols] = resultMat.shape
    confMat = zeros((rows,rows))
    acc = 0
    for r in range(rows):
      for c in range(cols):
        confMat[resultMat[r][c]][r] += 1
        if resultMat[r][c] == r:
          acc += 1

    return confMat, float(acc)/(rows*cols)
 

# loads data and computes a given number of random folds of cross validation
def randomFoldCrossValidation(numTrials = 10):
    dataDir = "../data"
    genres = ['classical','country','jazz','pop','rock','techno']
    
    
    #Step 1: load data into a list of list representation - each song is a dictionary
    print "Loading Data..."
    data = loadData(dataDir, genres)
    [cfm, acc] = processMusic(genres,data)
    for i in range(numTrials-1):
      newCfm, newAcc = processMusic(genres,data)
      cfm = cfm + newCfm
      acc = acc + newAcc
  
    print genres
    print cfm
    print "Overall Accuracy:",acc/float(numTrials)


# main program    
randomFoldCrossValidation(10)    
