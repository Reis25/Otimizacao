#include <bits/stdc++.h> 
#include <algorithm>
#include <vector> 

using namespace std; 
   
void colorEdges(int ptr, vector<vector<pair<int, int> > >& gra, 
                vector<int>& edgeColors, bool isVisited[]) 
{ 
    queue<int> q; 
    int c = 0; 
  
    set<int> colored; 
  
    //Verifica e já foi visitado 
    if (isVisited[ptr]) 
        return; 
  
    // Marcando o nó atual como visitado; 
    isVisited[ptr] = 1; 
  
    for (int i = 0; i < gra[ptr].size(); i++) { 
        if (edgeColors[gra[ptr][i].second] != -1) 
            colored.insert(edgeColors[gra[ptr][i].second]); 
    } 
  
    for (int i = 0; i < gra[ptr].size(); i++) { 
        if (!isVisited[gra[ptr][i].first]) 
            q.push(gra[ptr][i].first); 
  
        if (edgeColors[gra[ptr][i].second] == -1) { 
            //negativo
            while (colored.find(c) != colored.end()) 
  
                //cor
                c++; 
            edgeColors[gra[ptr][i].second] = c; 
  
            // adicionando dado na lista
            colored.insert(c); 
	//cout << c; 
            c++; 
        } 
    } 
  
// 	cout << "\n\n =================================================================== \n \n";
   // verificando se está vazia 
    while (!q.empty()) { 
        int temp = q.front(); 
        q.pop(); 
  
        colorEdges(temp, gra, edgeColors, isVisited); 
    } 
  
    return; 
} 
  

int main() 
{ 
    set<int> empty; 
  
    //Pares de vetores-> Grafo 
    vector<vector<pair<int, int> > > gra; 
	
    
    vector<int> edgeColors; 
    bool isVisited[100] = {0}; 
  
   
  	cout << "Digite o numero de vérices do grafo: \n";	
    	int ver;
   	cin >> ver;

   	
	cout << "Digite o numero de arestas  do grafo: \n";	
    	int edge;
   	cin >> edge;
   	 
     int color[edge] = {};
     //int count = 0
     int sum =0;
     int soma_total =0;
	
    gra.resize(ver); 
    edgeColors.resize(edge, -1); 

    for(int i=0; i <edge; i++){

	cout << "Digite a coordenada x e y \n";	
	int x, y;
	cin >> x >> y;


	gra[x].push_back(make_pair(y, i)); 
    	gra[y].push_back(make_pair(x, i)); 

	}
	
  colorEdges(0, gra, edgeColors, isVisited); 
  
    cout << "\n =================================================================== \n \n";
 	
    for (int i = 0; i < edge; i++){
        cout << "Vertice: " << i + 1 << " possui cor: " << edgeColors[i] + 1 << "\n";
	} 
    
   cout << "\n =================================================================== \n \n";

	for(int i =0; i <edge; i++){ 
	cout << "A cor: " << i + 1 << " Possui peso: " << i + 1 <<"\n"; 
	}   
 
     cout << "\n =================================================================== \n \n";
	
  sort(edgeColors.begin(), edgeColors.end());

    cout << "cores dos vértices ordenadas:\n"; 
    for (int i = 0; i < edge; ++i){
        cout << edgeColors[i] +1 << " ";
	}

    cout << "\n\n =================================================================== \n \n";
	
	// achar elementos repetidos: 
	int a = 0; 

	for (int i = 0; i < edge; ++i){
	
	if(edgeColors[i] == edgeColors[i+1]){

	sum = sum + edgeColors[i];
	cout << "Valor do edgeColors[i]: "<< edgeColors;
	cout << "Valor do sum: " << sum; 
	}
	else{
	color[a] = sum + edgeColors[i];
	sum = 0;
	a++;
	}
}
	
	// Calcular a soma com os respectivos pesos: 
	for (int i = 0; i < a; ++i){
	cout << "Vetor color["<< i <<"]= " << color[i] << "\n"; 
	soma_total = soma_total + color[i]*(i+1);
	}

	cout << "\n\n =================================================================== \n \n";

     cout << "Total de cores: " << a << "\n";
     cout << "Peso total: " << soma_total << "\n";

    return 0; 
} 

//auxilio: https://www.geeksforgeeks.org/edge-coloring-of-a-graph/
