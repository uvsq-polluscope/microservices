
# Mise en place et utilisation des micro services
Ce README a pour objectif de présenter la structure du projet et 

## Structure du répertoire Github
Le répertoire comporte les dossiers suivants:
- dataloader
- dataloaderGPS
- dataloader_smd
- classification
- preprocessing_gps
- preprocessing_time_series
- stop_move_detection
- post_processing

Les dossiers dataloader* permettent de récupérer des données de la base de données et de les ajouter à un topic qui est configuré. Les autres dossiers correspondent chacun à un micro-service.
Pour pouvoir utiliser une fonctionnalité, il faut s'assurer que [l'infrastructure](https://github.com/uvsq-polluscope/Infrastructure "Lien Github de l'infrastructure") est lancée et fonctionne correctement (ce qui est déjà le cas sur le serveur car l'infrastructure est déployée).

## Lancement d'un dataloader

Pour lancer un dataloader il suffit de se placer dans le dossier correspondant et d'exécuter la commande **python app.py**. Celui-ci va récupérer les données à partir de la requête SQL renseignée et les ajouter au topic défini.

## Lancement d'un micro-service

Pour lancer un micro-service il suffit de se placer dans le dossier correspondant et d'exécuter la commande **uvicorn main:app --reload**. L'API de ce micro-service va démarrer, pour lancer l'écoute des messages sur le topic renseigné il faut accéder à l'URL *IP/nom_du_microservice*.
Les microservices qui exposent des endpoints REST sont les suivants :
- http://192.168.33.124:8001/preprocessing : pour lancer le service preprocessing
- http://192.168.33.124:8002/preprocessing_gps : pour lancer le service preprocessing des données GPS
- http://192.168.33.124:8000/classification  : pour lancer la classification
- http://192.168.33.124:8004/stop_move_detection_algo : pour lancer l'algorithme stop move

### Scénario de lancement 

```
$>docker-compose up -d 
```
### Liberation des ressources 

```
 $> docker-compose down 
 ```
