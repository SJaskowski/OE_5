from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import random
from deap import tools
from deap import base
from deap import creator
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import  pandas as pd
import array as tmp_coef

pd.set_option('display.max_columns', None)
df=pd.read_csv("D:\Sprawozdania\PK\OE\proj4\data.csv",sep='\t', delimiter=',')

y=df['Status']
df.drop('Status',axis=1,inplace=True)
df.drop('ID',axis=1,inplace=True)
df.drop('Recording',axis=1,inplace=True)
df = df.apply( pd.to_numeric, errors='coerce' )
numberOfAtributtes= len(df.columns)



iris = df
mms = MinMaxScaler()
x_norm = mms.fit_transform(iris)
activation = ["identity", "logistic", "tanh", "relu"]
solver = ["lbfgs", "sgd", "adam"]
Wyniki = []
split = 5
cv = StratifiedKFold(n_splits=split)


Ilosc_cech=numberOfAtributtes
Ilosc_klas=6
neurons=15



def ParametersFeatures(icls):
    genome = list()
    for i in range(0, (Ilosc_cech*neurons+neurons*Ilosc_klas)+neurons+Ilosc_klas):
        genome.append(random.uniform(-1, 1))
    return icls(genome)


def ParametersFeatureFitness(individual):

    split = 5
    cv = StratifiedKFold(n_splits=split)
    estimator = MLPClassifier(hidden_layer_sizes=(neurons,), random_state=42, max_iter=200)
    final_result = 0
    for train, test in cv.split(x_norm, y):
        estimator.fit(x_norm[train], y[train])
        licznik = 0
        tmp_coefs1 = []
        for i in estimator.coefs_[0]:
            tmp_row = []
            for j in i:
                j = individual[licznik]
                tmp_row.append(j)
                licznik += 1
            tmp_coefs1.append(tmp_row)
        estimator.coefs_[0] = np.asanyarray(tmp_coefs1)
        tmp_coefs = []
        for i in estimator.coefs_[1]:
            tmp_row = []
            for j in i:
                j = individual[licznik]
                tmp_row.append(j)
                licznik += 1
            tmp_coefs.append(tmp_row)
        estimator.coefs_[1] = np.asanyarray(tmp_coefs)

        tmp_coefs = []
        for i in estimator.intercepts_[0]:
            i = individual[licznik]
            licznik += 1
            tmp_coefs.append(i)
        estimator.intercepts_[0] = np.asanyarray(tmp_coefs)

        tmp_coefs = []
        for i in estimator.intercepts_[1]:
            i = individual[licznik]
            licznik += 1
            tmp_coefs.append(i)
        estimator.intercepts_[1] = np.asanyarray(tmp_coefs)
        predicted = estimator.predict(x_norm[test])
        expected = y[test]
        accuracy = accuracy_score(expected, predicted)
        final_result = final_result + accuracy

    return final_result / split,


def mutationSVC(individual):
    i = random.randint(0,  (Ilosc_cech*neurons+neurons*Ilosc_klas)+neurons+Ilosc_klas-1)
    individual[i] = random.uniform(-1, 1)



sizePopulation = 25
probabilityMutation = 0.2
probabilityCrossover = 0.8
numberIteration = 50

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maksymalizacja
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register('individual', ParametersFeatures, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", ParametersFeatureFitness)
toolbox.register("select", tools.selLexicase)
toolbox.register("mate", tools.cxUniform, indpb=0.2)
toolbox.register("mutate", mutationSVC)
print("test1")
pop = toolbox.population(n=sizePopulation)
print("test2")
fitnesses = list(map(toolbox.evaluate, pop))
print("test3")
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit


x_avg = []
y_avg = []
z_avg = []

x_std = []
y_std = []
z_std = []

x_value = []
y_value = []
z_value = []

g = 0
numberElitism = 1
WholeAvg=0
while g < numberIteration:
 g = g + 1
 print("-- Generation %i --" % g)
 # Select the next generation individuals
 offspring = toolbox.select(pop, len(pop))
 # Clone the selected individuals
 offspring = list(map(toolbox.clone, offspring))
 listElitism = []
 for x in range(0, numberElitism):
    listElitism.append(tools.selBest(pop, 1)[0])
 # Apply crossover and mutation on the offspring
 for child1, child2 in zip(offspring[::2], offspring[1::2]):
 # cross two individuals with probability CXPB
    if random.random() < probabilityCrossover:
       toolbox.mate(child1, child2)
       # fitness values of the children
       # must be recalculated later
       del child1.fitness.values
       del child2.fitness.values
 for mutant in offspring:
   # mutate an individual with probability MUTPB
   if random.random() < probabilityMutation:
       toolbox.mutate(mutant)
       del mutant.fitness.values
   # Evaluate the individuals with an invalid fitness
   invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
   fitnesses = map(toolbox.evaluate, invalid_ind)
   for ind, fit in zip(invalid_ind, fitnesses):
       ind.fitness.values = fit
   print(" Evaluated %i individuals" % len(invalid_ind))
   pop[:] = offspring + listElitism

 # Gather all the fitnesses in one list and print the stats
 fits = [ind.fitness.values[0] for ind in pop]
 x_value = [ind[0] for ind in pop]
 y_value = [ind[1] for ind in pop]

 length = len(pop)
 mean = sum(fits) / length
 sum2 = sum(x * x for x in fits)
 std = abs(sum2 / length - mean ** 2) ** 0.5
 z_value=fits
 y_avg.append(mean)
 y_std.append(std)
 print(" Min %s" % min(fits))
 print(" Max %s" % max(fits))
 print(" Avg %s" % mean)
 print(" Std %s" % std)
 best_ind = tools.selBest(pop, 1)[0]
 print("Best individual is %s, %s" % (best_ind,
                                        best_ind.fitness.values))
  #
 print("-- End of (successful) evolution --")

for i in range (1,numberIteration):
  x_avg.append(i)
print("średnia Wartość: "+ str(sum(y_avg)/g))
fig=go.Figure()
#avg
fig.add_trace(go.Scatter(x=x_avg,y=y_avg,name="Średnia dokładność"))
fig.add_trace(go.Scatter(x=x_avg,y=y_std,name="Odchylenie standardowe"))
fig.update_layout(
    title="Średnia i odchylenie w zależności od iteracji",
    xaxis_title="Iteracja",
    yaxis_title="y",
    font=dict(
        family="Courier New, monospace",
        size=36,
        color="#7f7f7f"
    )
)
fig.show()


fig=px.scatter_3d(x=x_value,y=y_value,z=z_value)
fig.update_layout(
    title="Wartość funkcji celu",
    xaxis_title="X",
    yaxis_title="Y",
    font=dict(
        family="Courier New, monospace",
        size=22,
        color="#7f7f7f"
    )
)
fig.show()