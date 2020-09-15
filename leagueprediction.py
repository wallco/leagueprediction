# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('high_diamond_ranked_10min.csv')

# DEFINING RELEVANT RELATIVE FEATURES
# ALL FEATURES ARE CALCULATED IN RELATION TO BLUE SIDE (BLUE MINUS RED)

df['WardPlaceDiff']=df['blueWardsPlaced']-df['redWardsPlaced']
df['WardDestroyDiff']=df['blueWardsDestroyed']-df['redWardsDestroyed']
df['FirstBloodDiff']=df['blueFirstBlood']-df['redFirstBlood']
df['KillDiff']=df['blueKills']-df['redKills']
df['DeathDiff']=df['blueDeaths']-df['redDeaths']
df['AssistDiff']=df['blueAssists']-df['redAssists']
df['EliteMonsterDiff']=df['blueEliteMonsters']-df['redEliteMonsters']
df['DragonDiff']=df['blueDragons']-df['redDragons']
df['HeraldDiff']=df['blueHeralds']-df['redHeralds']
df['TowerDestroyDiff']=df['blueTowersDestroyed']-df['redTowersDestroyed']
df['AvgLevelDiff']=df['blueAvgLevel']-df['redAvgLevel']
df['MinionsDiff']=df['blueTotalMinionsKilled']-df['redTotalMinionsKilled']
df['JungleMinionsDiff']=df['blueTotalJungleMinionsKilled']-df['redTotalJungleMinionsKilled']
df['CSdiff']=df['blueCSPerMin']-df['redCSPerMin']
df['GPMdiff']=df['blueGoldPerMin']-df['redGoldPerMin']

relevant=['blueWins','WardPlaceDiff','WardDestroyDiff','FirstBloodDiff','KillDiff','DeathDiff',
                    'AssistDiff','EliteMonsterDiff','DragonDiff','HeraldDiff','TowerDestroyDiff',
                    'AvgLevelDiff','MinionsDiff','JungleMinionsDiff','blueGoldDiff','blueExperienceDiff',
                    'CSdiff','GPMdiff']

# SPLITTING DATA BETWEEN TRAINING AND TESTING AND SCALING

dados = df[relevant]

dados_embaralhados=dados.sample(frac=1, random_state = 143)

x = dados_embaralhados.loc[:,dados_embaralhados.columns!='blueWins'].values
y = dados_embaralhados.loc[:,dados_embaralhados.columns=='blueWins'].values

q = 7750

x_treino = x[:q,:]
y_treino = y[:q].ravel()

x_teste = x[q:,:]
y_teste = y[q:].ravel()


scaler = MaxAbsScaler()
scaler.fit(x_treino)

x_treino = scaler.transform(x_treino)
x_teste = scaler.transform(x_teste)


#BUILDING THE CLASSIFIER

classificador = KNeighborsClassifier(n_neighbors=5)

classificador = classificador.fit(x_treino, y_treino)

#-------------------------------------------------------------------------------
# TESTING THE CLASSIFIER WITH THE TRAINING SET
#-------------------------------------------------------------------------------

y_resposta_treino = classificador.predict(x_treino)

#-------------------------------------------------------------------------------
# OBTAINING THE TEST SET PREDICTIONS
#-------------------------------------------------------------------------------

y_resposta_teste = classificador.predict(x_teste)

#-------------------------------------------------------------------------------
# VERIFYING ACCURACY
#-------------------------------------------------------------------------------

print ("\nPERFORMANCE INSIDE TRAINING SET\n")

total   = len(y_treino)
acertos = sum(y_resposta_treino==y_treino)
erros   = sum(y_resposta_treino!=y_treino)

print ("Total samples: " , total)
print ("Correct predictions:" , acertos)
print ("Wrong predictions: " , erros)

acuracia = acertos / total

print ("Accuracy = %.1f %%" % (100*acuracia))

print ("\nPERFORMANCE OUTSIDE TRAINING SET\n")

total   = len(y_teste)
acertos = sum(y_resposta_teste==y_teste)
erros   = sum(y_resposta_teste!=y_teste)

print ("Total samples: " , total)
print ("Correct predictions:" , acertos)
print ("Wrong predictions: " , erros)

acuracia = acertos / total

print ("Accuracy = %.1f %%" % (100*acuracia))

#-------------------------------------------------------------------------------
# VERIFYING ACCURACY FOR DIFFERENT K VALUES
#-------------------------------------------------------------------------------

print ( "\n  K TRAINING  TEST")
print ( " -- ------ ------")

for k in range(15,35):

    classificador = KNeighborsClassifier(
        n_neighbors = k,
        weights     = 'uniform',
        p           = 1
        )
    classificador = classificador.fit(x_treino,y_treino)

    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste  = classificador.predict(x_teste)
    
    acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)
    acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)
    
    print(
        "%3d"%k,
        "%6.1f" % (100*acuracia_treino),
        "%6.1f" % (100*acuracia_teste)
        )
    
    


















    