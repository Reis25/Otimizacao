/*
 * SampleDecoder.cpp
 *
 *  Created on: Jan 14, 2011
 *      Author: rtoso
 */

// Criar G como global 
// extern vector<int> G; 
#include "SampleDecoder.h"

SampleDecoder::SampleDecoder() { }

SampleDecoder::~SampleDecoder() { }

// Runs in \Theta(n \log n):
double SampleDecoder::decode(const std::vector< double >& chromosome) const {
	std::vector< std::pair< double, unsigned > > ranking(chromosome.size());

	for(unsigned i = 0; i < chromosome.size(); ++i) {
		ranking[i] = std::pair< double, unsigned >(chromosome[i], i);
	}

	// Here we sort 'permutation', which will then produce a permutation of [n] in pair::second:
	std::sort(ranking.begin(), ranking.end());
	

// permutation são os índices dos vértices imunizados; 
	vector<int> permutation; 
	for( int i =0; i <K; i++){ 
	permutation.push_back(ranking[i].second);
	}
	double custo = calcula_custo(G*, permutation);
	return custo;
	
}
