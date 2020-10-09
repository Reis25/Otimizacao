#include "busca_decode.h"
#include <vector>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <queue>
#include<list>
#include <fstream>



using namespace std; 

//extern  vector< vector<int> > G;
extern int K_vertices = 1;

busca_decode::busca_decode() { }
busca_decode::~busca_decode() { }

double busca_decode::decode(const std::vector< double > & chromosome) const{
  std::vector< std::pair< double, unsigned > > ranking(chromosome.size());

for(unsigned i = 0; i < chromosome.size(); ++i) {
    ranking[i] = std::pair< double, unsigned >(chromosome[i], i);
  }

  std::sort(ranking.begin(), ranking.end());
  
  // vetor booleano com n posições falsas (p cd vértice) 
  std::vector<bool> permutation(chromosome.size(), false); 

  for( int i =0; i <K_vertices; i++){ 
  permutation[ranking[i].second] = true;
  }
  double custo = calc_cust(permutation, 1, "out");

   
  return custo;
  
}


double busca_decode::calc_cust(std::vector <bool>& permutation, int s, const string& instance) const{

  std::list<unsigned>:: iterator iteratorBrkga;
  std::string inputs,file;
  std::fstream readFile;

  vector<unsigned> routeAgv;
  queue<int> q;
 
  unsigned v0,v1;
  double conver;

  list<unsigned>* list_vertices;
  list_vertices = new list<unsigned> [16];
 
  std::vector<unsigned> listOfPositions;

  readFile.open("input_file.txt");
    if(readFile){
        while (getline(readFile,inputs))
        {
            conver = (double)(stoi(inputs));
            listOfPositions.push_back(conver);
        }
  readFile.close();

    }
    else{
        std::cout<< "incorrect name file or do not exists";
    }

  for(int i=0;i<16;i++)

  permutation[i]=false;

   for(iteratorBrkga = list_vertices[0].begin();iteratorBrkga!= list_vertices[0].end();iteratorBrkga++){
                if(!permutation[*iteratorBrkga]){
                    permutation[*iteratorBrkga]=true;
                    q.push(*iteratorBrkga);
                    routeAgv.push_back(*iteratorBrkga);
                }
            }
 
  for(unsigned i=0; i< routeAgv.size();i++){
            cout<<routeAgv[i]<<",";
          }

  int count = 0;
  
  /*
  
  permutation[s] = true;
  
  q.push_back(s);
  
  while(!q.empty()){
 
  s = q.front();

  count++; 

  //cout << s << " "; 

  q.pop_front();

    int i = 0;

    for(int i = 0; i < K_vertices; i++){
      
      if(!permutation[i]){

      permutation[i] = true;

      q.push_back(i);
      }
    }  
  
   8
  
}*/
return count;
}


/*
double busca_decode::calc_cust_graph(std::vector <bool> & permutation) const{
  int total = 0;

  int tamanho_vetor = sizeof(q.size())/sizeof(int);

  for(int i =0; i < tamanho_vetor; i++){

  if(!permutation[i]){

  int s = calc_cust(permutation);
  
         total += s*s; 
  
  }
}
  return total/permutation.size();
}
*/


 