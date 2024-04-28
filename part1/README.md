# Partie 1: Highway Environment

Cette partie du projet se concentre sur l'implémentation et l'expérimentation d'un agent de Reinforcement Learning (RL) pour résoudre l'environnement Highway.

## Notebooks

- **dqn_highway.ipynb:** Ce notebook contient le code initial de notre modèle DQN, basé sur le code du TP4 mais corrigé et adapté à notre problème. Malheureusement, les performances de ce modèle sont médiocres. Les poids de cet agent entraîné sur 3000 itérations sont sauvegardés dans le fichier `dqn_agent_3000.pth`.

- **dqn_highway_improved.ipynb:** Ce notebook contient le code pour un modèle amélioré, utilisant un CNN. C'est également avec ce modèle que l'agent "NoOffRoad" est entraîné. Les poids de ce modèle sont sauvegardés dans les fichiers `dqn_agent_modif4_10000.pth` et `dqn_agent_modif5_20000.pth` pour des entraînements respectivement sur 10 000 et 20 000 itérations. De plus, les poids du modèle où l'agent arrête l'entraînement lorsqu'il sort de la route sont sauvegardés dans le fichier `dqn_agent_modif_noOffRoad_10000.pth`.

- **dqn_highway_complex_cnn.ipynb:** Ce notebook contient une tentative de modèle plus complexe. Les poids de ce modèle sont sauvegardés dans le fichier `dqn_agent_modif_complex_cnn_10000.pth`.

## Poids Enregistrés

Les poids des modèles entraînés et leurs performances associées sont sauvegardés dans des fichiers `.pth`. Les noms des fichiers indiquent clairement le nombre d'itérations et les caractéristiques du modèle correspondant.

## Graphiques

Dans ce dossier, vous trouverez également des images des graphiques montrant la perte (loss) de certains entraînements de ces modèles. Les noms de ces images sont explicites pour identifier l'entraînement correspondant.
