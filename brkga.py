#%%
import numpy as np
import pandas as pd
from prediction import predict_model, metrics_by_model

class GeneticSelection:
    ''' Genetic algorithm to feature selection
    
    Attributes
    ----------
    random : instance of numpy random state
    
    n_population : int
        Number of individuals on the population
        
    n_generations : int
        Max number of generations
        
    tournament_size : int
        Size of tournament selection method
        
    crossover_proba : float
        Probability of changing chromossome points in uniform crossover
        
    mutation_proba : float
        Simple probability on mutating a chromossome in a individual
        
    n_gen_no_change : int
        Max number of generations without change it's best value
        
    scoring : str
        Metric to be optimized
        ['accuracy', 'average_precision', 'f1', 'ppv', 'tpr', 'tnr', 'roc_auc']
    
    '''
    def __init__(self, random_state=0, scoring='roc_auc', n_population=50, n_generations=100,
                 elite_solutions_size=10, n_gen_no_change=10, elite_inheritance_proba=0.75, mutants_solutions_size=35):
        self.random = np.random.RandomState(random_state) if isinstance(random_state, int) else random_state
        self.n_population = n_population
        self.n_generations = n_generations
        self.n_gen_no_change = n_gen_no_change
        self.scoring = scoring
        self.elite_solutions_size = elite_solutions_size
        self.mutants_solutions_size = mutants_solutions_size
        self.elite_inheritance_proba = elite_inheritance_proba

    def fit(self, base, model):
        ''' Function that starts the genetic feature selection        
        Parameters
        ----------
        base : tuple (<PANDAS DF>, <PANDAS SERIES>)
            where the first value is a pandas dataframe with non class features of a
            dataset, the second value correspond to class value for each sample.
        model : classification model instance
        
        Returns : list
            Each element on the list is a tuple (<FEATURES>, <fitness_score>)
            the first one is a binary list of features to be used (1) or not (0)
            the second tuple value is the fitness value of the features list

	passando os parametros do grafo
	passamos o grafo
        '''
	 matrix_graph = [[0 for x in range(10) for y in range(10)]
	
        (X, _) = base

        if type(X) is type(pd.DataFrame()):
            individual_size = len(X.iloc[0, :].values)
        else:
            individual_size = len(X[0])

        self.individual_size = individual_size
        self.individual_fitness_map = {}

        print("----- Starting population -----")
        initial_population = self.generate_individuals(base, model, self.n_population)
        final_population = self.generations(base, model, initial_population)
        
        return final_population

    def generations(self, base, model, population):
        ''' Applies generations over the initial population, returns the final population
        of individuals
        
        Parameters
        ----------
        base : tuple (<PANDAS DF>, <PANDAS SERIES>)
            where the first value is a pandas dataframe with non class features of a
            dataset, the second value correspond to class value for each sample.
        model : classification model instance
        population : list
            a list of binary lists, individuals to be optimized through the generations
        
        Returns
        -------
        population : list
            a list of binary lists, individuals optimized through the generations
        '''
        gen_no_change = 0
        population.sort(key=self.fitness_score, reverse=True)
        best_individual = population[0]
        for generation in range(self.n_generations):
            print("--- Running generation %d of %d ---"%(generation+1, self.n_generations))
            elite, non_elite = self.selection(population)
            new_mutants = self.generate_individuals(base, model, self.mutants_solutions_size)
            new_children_size = self.n_population - len(elite) - len(new_mutants)
            new_children = self.breeding(best_individual, population[1:-1], new_children_size)
            population = elite + new_children + new_mutants
            assert len(population) == self.n_population
            for individual in population:
                self.evaluation(base, model, individual)
            population.sort(key=self.fitness_score, reverse=True)
            new_best_individual, new_best_fitness = (population[0], 
                                                     self.fitness_score(population[0]))
            print("Best individual: %s | Score: %s"%(new_best_individual, new_best_fitness))
            if not str(np.array(new_best_individual)) == str(np.array(best_individual)):
                best_individual = new_best_individual
                gen_no_change = 0
            else:
                gen_no_change = gen_no_change + 1
            if gen_no_change > self.n_gen_no_change:
                break
        return population

    def selection(self, population):
        ''' Applies tournament method to select individuals from the population        
        '''
        selected_individuals = population[0:self.elite_solutions_size]
        non_selected_individuals = population[self.elite_solutions_size:-1]
        return selected_individuals, non_selected_individuals

    def crossover(self, parent_elite, parent_other):
        ''' Applies uniform crossover method over two individuals to generate
        other two individuals
        '''
        child = []
        for i in range(len(parent_elite)):
            if self.random.rand() > self.elite_inheritance_proba:
                child.append(parent_other[i])
            else:
                child.append(parent_elite[i])
        return child

    def generate_individuals(self, base, model, num_individuals):
        population = []
        while len(population) < num_individuals:
            individual = self.random.rand(self.individual_size)
            self.evaluation(base, model, individual)
            population.append(individual)
        return population

    def breeding(self, parent_a, parent_candidates, children_size):
        ''' Creates children individuals through parents crossovers and mutation of its childs
        
        Parameters
        ----------
        parents : list
            a list of binary lists
            
        Raises
        ------
        'Parents number exception' when the parents number is lower than 2
        
        Returns
        -------
        children : list
            a list of binary lists of individuals generated by crossovers of the parents 
            individuals
        '''
        n_parents = len(parent_candidates)
        children = []
        while len(children) < children_size:
            p1 = self.random.randint(0, n_parents)
            child = self.crossover(parent_a, parent_candidates[p1])
            children.append(child)
        return children

    def fitness_score(self, individual):
        ''' Function to get the fitness score, works through a individual-fitness map
        
        Parameters
        ----------
        individual : list
        
        Raises
        ------
        'Fitness not calculated' exception when the individual has not a related 
        evaluation on map
        
        Returns
        -------
        fitness : float
        '''
        try:
            features = self.encodeThreshold(individual)
            fitness = self.individual_fitness_map[str(features)]
        except:
            raise Exception('Fitness not calculated')
        return fitness


    def evaluation(self, base, model, individual):
        ''' Function to calculate the fitness score of a individual and save it in a map
        
        Parameters
        ----------
        base : tuple (<PANDAS DF>, <PANDAS SERIES>)
            where the first value is a pandas dataframe with non class features of a
            dataset, the second value correspond to class value for each sample.
        model : classification model instance
        individual : list
        
        Raises
        ------
        'Fitness could not be calculated' exception when the individual could not be calculated
        
        Returns
        -------
        individual : list
        '''
        features = self.encodeThreshold(individual)
        if str(features) in self.individual_fitness_map.keys():
            fitness = self.individual_fitness_map[str(features)]
        else:
            fitness = self.evaluate(base, model, features)
            self.individual_fitness_map[str(features)] = fitness
        return individual

    def encodeThreshold(self, features):
        return [ 1 if feature > 0.5 else 0 for feature in features ]

    def evaluate(self, base, model, features):
        ''' Function to calculate the fitness score of a individual 
        
        Parameters
        ----------
        base : tuple (<PANDAS DF>, <PANDAS SERIES>)
            where the first value is a pandas dataframe with non class features of a
            dataset, the second value correspond to class value for each sample.
        model : model classifier instance
        individual : list
    
        
        Returns
        -------
        score : float
        '''
        predictions = predict_model(base, model, features=features)
        score = metrics_by_model(predictions)[self.scoring].mean()
        return score

