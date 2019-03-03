#!/usr/bin/env python2.7
#------------------------------------------------------------#
# file: kmeans.py                                            #
#                                                            #
# description:                                               #
#   takes a CSV file and writes cluster labels to binary..   #
#                                                            #
# usage:                                                     #
#       ./kmeans.py [inputFileName ] [k ]                    #
# author: A. Richardson, M.Sc.                               #
#------------------------------------------------------------#

import os
import sys
import math
import time
import numpy
import struct
import random

random.seed(time.clock());

argv = sys.argv
if( len ( argv ) < 3 ):
  print("kmeans.py: usage:\n\t./kmeans [infile] [k]\n");
  sys.exit(1);
inFileName = (sys.argv[1]).strip();
if( not( os.path.exists( inFileName))):
  print("Error: could not open input file :"+str(inFileName));
  sys.exit(1);

#------------------------------------------------------------#
# description:                                               #
#     data reading section.                                  #
#------------------------------------------------------------#

def split(x): return x.strip().split(',');

lines = open(inFileName).readlines();
nRecords = len(lines)-1;      #how many measurement vectors?
fieldNames = split(lines[0]);
nFields = len(fieldNames);  #how many fields?
lines = lines[ 1:]; #lose the first line/ header..

#init. a numpy format container...
data = numpy.empty([nRecords, nFields]); #numpy data format.
if( data.shape[0] != nRecords or data.shape[1] != nFields):
  print('Error: data.shape != nRecords, nFields'); sys.exit(1);

#------------------------------------------------------------#
# description:                                               #
#     check data integrity while populating the container..  #
#------------------------------------------------------------#

nNan = 0;
for i in range(0,len(lines)):
  row = split(lines[i]);
  if(len(row)!=nFields):
    print(str(row)+'Error: row length '+str(len(row))+'!='+str(nFields));
    sys.exit(1);
  for j in range(0,len(row)):
    dataValue = str(row[j]).strip();
    if( dataValue=='nan'):
      Nan +=1;
      data[i,j]=float('nan');
    try:
      data[i,j] =float(dataValue);
    except:
      data[i,j] =float('nan'); nNan +=1;
      print('Warning: nan at row i='+str(i+1)+ ' (1-indexed), col='+str(j+1));
print('There are '+str(nRecords)+' measurement vectors.');
if( nNan >0): print('Warning: '+str(nNan) + ' instances-of-NaN detected. ');

#------------------------------------------------------------#
# description:                                               #
#     output CSV to make sure data is intelligible..         #
#------------------------------------------------------------#

def writeCSV():
  f = open('dataArray.csv','wb');
  f.write( ','.join(fieldNames)+'\n');
  for i in range(0,nRecords):
    for j in range(0,nFields):
      if(j>0):
        f.write(',');
      f.write(str(data[i,j]));
    f.write('\n')
  f.close();


#------------------------------------------------------------#
# description:                                               #
#     find the max and min of the individual features..      #
#------------------------------------------------------------#

dataMax = numpy.zeros( nFields);
dataMin = numpy.zeros( nFields);

for j in range(0,nFields):
  dataMax[j] = sys.float_info.min;
  dataMin[j]= sys.float_info.max;
  for i in range(0,nRecords):
    dij = data[i,j];
    if( dij > dataMax[j]):
      dataMax[j]=dij;
    if( dij < dataMin[j]):
      dataMin[j]=dij;

dataScale = numpy.zeros(nFields);

for j in range(0,nFields):
  print('var: '+fieldNames[j]+' min '+str(dataMin[j])+' max '+str(dataMax[j]))
  mM = dataMax[j]-dataMin[j];
  dataScale[j] = 1./(mM);
  dataScale[j] = 1.; dataMin[j] = 0.;

#------------------------------------------------------------#
# description:                                               #
#     kmeans clustering section.. cluster variable setup..   #
#------------------------------------------------------------#

nClust = 5; #number of clusters to initialize..
maxIter = 100;  #number of iterations for cluster optimization..
try:
  nClust = int( sys.argv[2]);
except:
  nClust = 5;


clusterCentres = [ ]; #line index (i) for cluster representative
for i in range(0,nClust): clusterCentres.append( -1);

currentLabel = [ ];#current label (cluster index) for a point...
for i in range(0,nRecords): currentLabel.append( -1);

#(key,value)=(cluster index, list of members (pt. indices i))
clusterMembers ={ };

#------------------------------------------------------------#
# description:                                               #
#     find the 'most central' element of a cluster...        #
#------------------------------------------------------------#

def mostCentralElementForClusterJ( j):
  global data, clusterMembers;
  nRecords, nFields = data.shape;
  nObs = 0.;
  myMean = [ ] ;
  for i in range(0,nFields):  myMean.append(0.);
  myMembers = clusterMembers[j];
  nMembers  = len( myMembers);
  if( nMembers ==0):
    print('Error: no members in class '+str(j));
    return -1;
  for i in range(0,nMembers):
    myMember = myMembers[i];
    for m in range(0,nFields):
      myMean[m]+= (data[myMember, m] - dataMin[m])*(dataScale[m]);
    nObs +=1.;
  for m in range(0,nFields):
    myMean[m] = myMean[m]/nObs;
  #find the member closest to the mean..
  #feedback between elements, vs. representatives..
  assignI = 0;
  minD = math.sqrt(math.sqrt( math.sqrt( sys.float_info.max )));
  minI = float('nan');
  for i in range(0,nMembers):
    d = 0.;
    myMember = myMembers[i]
    for m in range(0,nFields):
      d += math.pow( myMean[m] - ((data[ myMember, m]-dataMin[m])*dataScale[m]),2.);
    d = math.sqrt( d );
    if( assignI ==0 or  (d<minD and not(math.isnan(d))) ):
      minD = d; minI = myMember; assignI+=1;
  #assign the 'most central' element as the representative..
  if( minI==float('nan')):
    print('Error: no centre rep for clusterJ: j='+str(j));
    sys.exit(1);
  return(minI);

#------------------------------------------------------------#
# description:                                               #
#     function to find the nearest cluster centre, relative  #
#       to a given observation                               #
#------------------------------------------------------------#

def labelOfCentreNearestToAnObservationAtIndexI( i):
  global data, clusterCentres;
  nRecords, nFields = data.shape;
  nClust = len(clusterCentres);
  minD = math.sqrt(math.sqrt( math.sqrt( sys.float_info.max )));
  assignI = 0;
  minI = -1;
  for j in range(0,nClust): #don't consider empty clusters..
    if(clusterCentres[j] == -1): continue;
    d = float(0.);
    centreI = clusterCentres[j];
    for k in range(0,nFields):
      dd = (data[centreI, k]  - data[i, k])*(dataScale[k]);
      d+= dd*dd;
    d = math.sqrt(d);
    if( assignI==0 or (d<minD and not(math.isnan(d)))):
      minI = j; minD = d; assignI +=1;
  if(minI==float('nan') or str(minI)=='nan'):
    print('Error: no centre nearest to obs. i='+str(i));
    sys.exit(1);
  return( minI);

#------------------------------------------------------------#
# description:                                               #
#     seed the clustering...                                 #
#------------------------------------------------------------#

#seed the simulation..
for i in range(0,nRecords): #randomly label the points:
  currentLabel[i] =random.randint(0, nClust-1);

def writeLabels():
  global currentLabel
  f=open("label.bin","wb");
  for i in range(0,len(currentLabel)):
    d = float(currentLabel[i]);
    s = struct.pack('f', d);
    f.write( s);
  f.close();


#------------------------------------------------------------#
# description:                                               #
#   run the (unsupervised) algorithm...                      #
#------------------------------------------------------------#

lastClusterCentres = str(clusterCentres);

#for a number of iterations:
for k in range(0,maxIter):
  #calculate class membership lists:
  clusterMembers={ };
  for i in range(0,nClust):
    clusterMembers[i] = [];
  for i in range(0,nRecords):
    myLabel = currentLabel[i]
    clusterMembers[myLabel].append(i);
  print('1) clusterMembers: '+str(clusterMembers))
  #for each class, find the 'most central' element..
  for j in range(0,nClust):
    clusterCentres[j] = mostCentralElementForClusterJ(j);
    if( clusterCentres[j] =='nan' or clusterCentres[j]==float('nan')):
      print('Error: current label is nan for clusterCentre.');
      sys.exit(1);
  print('2) clusterCentres:'+str(clusterCentres))
  if( str(clusterCentres)==lastClusterCentres):
    print('cluster centres did not move after '+str(k+1)+' iterations.. done.');
    writeLabels();
    sys.exit(1);
  lastClusterCentres = str(clusterCentres);
  #label a point in terms of the nearest 'cluster centre':
  for i in range(0,nRecords):
    newLabel = labelOfCentreNearestToAnObservationAtIndexI(i);
    if(str(newLabel)=='nan' or newLabel==float('nan')):
      print('Error: current label is nan: '+str(i));
      sys.exit(1);
    currentLabel[i] = newLabel;
  print('1) currentLabels:'+str(currentLabel));
  print('-----------------------------------------------------------------');

printLabels();
