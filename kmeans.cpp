/* kmeans.cpp arichardson

Takes a CSV file and writes cluster labels to binary. This version not well tested.
Could be updated for larger data (e.g., long int indices, and more efficient data
structures, etc.) */

#include<stdlib.h>
#include<stdio.h>
#include<vector>
#include<sstream>
#include<string>
#include<iostream>
#include<fstream>
#include<limits>
#include<math.h>
#include<string>
#include<sstream>
using namespace std;

int nClust, nFields, nRecords;
float * datMax;
float * datMin;
float * datScale;
float * dat;
float * myMean;
int * clusterCentres;
int * currentLabel;
vector< vector<int> > clusterMembers;
std::string lastClusterCentres;

void cleanUp(){
  free(dat);
  free(datMin);
  free(datMax);
  free(datScale);
  free(clusterCentres);
  free(currentLabel);
  free(myMean);
}

/*convert int to string*/
string itos( int i){
  std::string number("");
  std::stringstream strstream;
  strstream << i;
  strstream >> number;
  return number;
}

/* split a string (a-la python) */
static vector<string> split(string s, char delim){
  vector<string> x;
  long int N = s.size(); if( N==0) return x;
  istringstream iss(s); string tok;
  while(getline(iss,tok,delim)) x.push_back(tok);
  return x;
}

/* allocate a float array.. */
float * alloc( int n ){
  float * dat = (float *)(void *)malloc( sizeof(float) * n);
  memset(dat, '0', sizeof(float)*n);
  return (dat);
}

/* allocate an int array.. */
int * ialloc( int n){
  /* wish we had variable precision arithmetic.. */
  int * dat = (int *)( void *)malloc(sizeof(int) * n);
  memset(dat, '0', sizeof(int)*n);
  return (dat);
}

/* find most central element for a cluster with index j */
int mostCentralElementForClusterJ(int j ){
  float nObs = 0.;
  int i;
  for(i = 0; i < nFields; i++){
    myMean[i] = 0.;
  }
  vector<int> * myMembers = &(clusterMembers[j]);
  if(myMembers->size() == 0){
    cout <<"Error: no members in class:"<<j<<endl;
    return -1;
  }
  vector<int>::iterator it;
  for(it = myMembers->begin(); it != myMembers->end(); it++){
    int myMember = *it;
    for(i = 0; i < nFields; i++) myMean[i] += (dat[myMember*nFields + i] - datMin[i]) * (datScale[i]);

    nObs += 1.;
  }
  for(i = 0; i < nFields; i++){
    myMean[i] = myMean[i] / nObs;
  }

  /* find the member closest to the mean.. feedback between clusters and reps.. */
  int assignI = 0;
  float minD =  (float)std::numeric_limits<int>::max();
  int minI = -1;

  for(it = myMembers->begin(); it != myMembers->end(); it++){
    float d = 0.;
    int myMember = *it;
    for(i = 0; i < nFields; i++){
      d += pow( myMean[i] - ((dat[myMember*nFields + i] - datMin[i]) * datScale[i]), 2.);
    }
    d = sqrt(d);
    if(assignI == 0 || ( d < minD && !isnan(d))){
      minD = d;
      minI = myMember;
      assignI++;
    }
  }
  if(minI == -1){
    cout<<"Error: no centre rep for clusterJ:"<<j<<endl;
    exit(1);
  }
  return(minI);
}

/* find label of the centre nearest to the given observation at index i.. */
int labelOfCentreNearestToAnObservationAtIndexI(int i){
  float minD =(float)std::numeric_limits<int>::max();
  int assignI = 0;
  int minI = -1;
  int j;
  for(j = 0; j < nClust; j++){
    if(clusterCentres[j] == -1) continue;
    float d = 0.;
    int centreI = clusterCentres[j];
    int k;
    for(k = 0; k < nFields; k++){
      float dd = (dat[centreI*nFields + k]-dat[i * nFields + k]) * datScale[k];
      d += dd * dd;
    }
    d = sqrt(d);
    if(assignI == 0 || (d < minD && !isnan(d))){
      minI = j;
      minD = d;
      assignI++;
    }
  }
  if(minI == -1){
    cout << "Error: no centre nearest to obs. i=" << i << endl;
    exit(1);
  }
  return minI;
}

void writeLabels(){
  FILE * f = fopen("label.bin", "wb");
  if(!f){
    cout<<"Error: could not open label.bin for writing."<<endl;
    exit(1);
  }
  int i;
  for(i = 0; i < nClust; i++){
    fwrite(&currentLabel[i], 1, sizeof(int), f);
  }
  fclose(f);
}

int main(int argc, char ** argv){
  if(argc < 2){
    printf("kmeans.cpp: usage:\n\t./kmeans [infile] [k]\n");
    exit(1);
  }
  std::ifstream f;
  f.open(argv[1]);
  if(!f){
    cout<<"Error: could not open file:" << argv[1] << endl;
    exit(1);
  }

  /* number of clusters to determine.. */
  nClust = 3;
  if(argc >= 3){
    nClust = atoi(argv[2]);
  }

  /* maximum number of iterations.. */
  int maxIter = 100;

  /* fixed line size is bad.. */
  char line[4096];
  std::vector<string> lines;
  while(f.getline(line, 4096)){
    lines.push_back(std::string(line));
  }
  vector<string>::iterator it, i;
  nFields = nRecords = 0;
  long int li = 0;
  datMax = datMin = datScale = dat = NULL;
  li=-1;

  /* read names of fields from first line.. */
  vector<string> fieldNames;
  for(it = lines.begin(); it != lines.end(); it++){
    li++;
    if(it == lines.begin()){
      fieldNames =  split(*it, ',');
      for(i = fieldNames.begin(); i != fieldNames.end(); i++){
        /* count the number of fields.. */
        nFields += 1;
      }
      nRecords = lines.size() -1;
      dat = alloc(nFields * nRecords);
      datScale = alloc(nFields);
      datMin = alloc(nFields);
      datMax = alloc(nFields);
      int j;
      for(j = 0; j < nFields; j++){
        /* initialize max/min calculation.. */
        datMin[j] =   (float)std::numeric_limits<int>::max();
        datMax[j] = - (float)std::numeric_limits<int>::max();
      }
    }
    else{
      vector<string> dataLine = split(*it, ',');
      if(dataLine.size()!= nFields){
        cout << "Error: line "<< li <<" had incorrect # of fields.\n";
        exit(1);
      }
      else{
        /* calculate max and min of each field, for each record.. */
        int j = 0;
        for(i = dataLine.begin(); i != dataLine.end(); i++){
          float d = std::atof((*i).c_str());
          if(d < datMin[j]) datMin[j] = d;
          if(d > datMax[j]) datMax[j] = d;
          dat[(li - 1) * nFields + j++] = d;
        }
      }
    }
  }
  int j;
  for(j = 0; j < nFields; j++){
    datScale[j] = 1. / (datMax[j] - datMin[j]);
  }
  cout << endl;
  for(j = 0; j < nRecords; j++){
    int k;
    for(k = 0; k < nFields; k++){
      cout << dat[j * nFields + k] << ",";
    }
    cout << endl;
  }

  /* set up clustering variables.. */
  clusterCentres = ialloc(nClust);
  for(j = 0; j < nClust; j++){
    clusterCentres[j] = -1;
  }
  currentLabel = ialloc(nRecords);
  for(j = 0; j< nRecords; j++){
    currentLabel[j] = -1;
  }

  /* list of members (point indices) for each cluster.. */
    for(j = 0; j < nClust; j++){
      vector<int> v;
      clusterMembers.push_back(v);
    }

  /* average (for a given cluster).. */
  myMean = alloc(nFields);

  //------------------------------------------------------------//
  //    seed the clustering..                                   //
  //------------------------------------------------------------//
  srand(time( NULL));
  for(j = 0; j < nRecords; j++){
    currentLabel[j] = (rand() % nClust);
  }

  //------------------------------------------------------------//
  //    run the unsupervised algorithm..                        //
  //------------------------------------------------------------//
  int k; lastClusterCentres = "";
  for(k = 0; k < maxIter; k++){
    cout << "Iter"<< k <<endl;
    /* calculate class membership lists.. */
    for(j = 0; j < nClust; j++){
      clusterMembers[j].clear();
    }

    for(j=0; j< nRecords; j++){
      int myLabel = currentLabel[ j ];
      cout << " push j: " << j << " myLabel " << myLabel << endl;
      clusterMembers[myLabel].push_back(j);
    }
    for(j = 0; j < nClust; j++){
      vector<int> * myMembers = &clusterMembers[j];
      vector<int>::iterator it;
      for(it = myMembers->begin(); it != myMembers->end(); it++){
        cout << *it << ",";
      }
      cout << endl;
    }
    std::string currentClusterCentres("");
    /* for each class, find 'the most central' element.. */
    for(j = 0; j < nClust; j++){
      clusterCentres[j] = mostCentralElementForClusterJ(j);
      if(clusterCentres[j] == -1){
        cout << "Error: current label is -1 for clusterCentre: j=" << j << endl;
        exit(1);
      }
      currentClusterCentres += itos(clusterCentres[j]) + std::string(",");
    }
    if(lastClusterCentres != currentClusterCentres){
      cout << "Cluster centres did not move after " << (k + 1) << " iterations." << endl;
      writeLabels();
      cleanUp();
      exit(1);
    }
  }

  //------------------------------------------------------------//
  //    clean up..                                              //
  //------------------------------------------------------------//
  cleanUp();

  return 0;
}