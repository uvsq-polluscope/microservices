
# Rapport : Approche micro-services pour l&#39;automatisation de pipeline de fouille de données

  

Par : Mohamed Ramdane Debiane, El Mamoun Affane-Aji, Julien Daurat, Amir Hammad, Julien Jacquet, Khaoula Hassoune, Mamadou Ndiaye, Sidi Mohamed Hicham Zekri, Séverine Selaquet, Kamilia Loualia, Abdellatif Missoumi.

  

##

  
  

## Sommaire

  

 1. **Sommaire**
    
      
   
 2. **Introduction**


 1. **Démarche du projet**

    
      

	 1. **Phase de recherche**

    
      

	 1. **Phase d&#39;apprentissage et découverte**

    
      

	 1. **Phase de planification de l&#39;architecture**

    
      
 

	 1. **Mise en Place**

    
      
    

 1. **Architecture du système**

    
      

	 1. **Détails de l&#39;infrastructure**

    
      
    

		 1. **Apache Zookeeper**

    
      
    
    

		 1. **Apache Kafka**

    
      
    
  

		 1. **Kafka Connect**

    
      

		 1. **Schema-Registry**

    
      
    

		 1. **Kafka-UI**

    
      
    
   

		 1. **Bases de données PostgreSQL**

    
      
 

	 1. **Détails pour la partie Timeseries**

    
      
    

	 1. **Détails pour la partie GPS**

    
      
  

		 1. **Preprocess GPS :**

    
      
   

		 1. **Stop move detection :**

    
      
    

 1. **Conclusion**

    
      

 1. **Bibliographie**

##

  
  

## Introduction

  

Polluscope est un observatoire participatif pour la surveillance de l&#39;exposition individuelle à la pollution de l&#39;air en lien avec la santé. C&#39;est un consortium qui regroupe : le laboratoire DAVID, AirParif, CEREMA, EIVP, iPLESP, Irenav et LSCE.

  

Des volontaires portent des capteurs de qualité de l&#39;air sur eux, et effectuent des mesures à intervalles réguliers. Les volontaires renseignent le lieu où ils ont pris les mesures, par exemple : dans les transports, chez eux, etc. Les capteurs permettent de mesurer les concentrations de différents gaz dans l&#39;air, ces mesures sont également géotaguées.

  

Les mesures sont envoyées sur une base de données PostgreSQL, et elles sont ensuite traitées avec différentes méthodes de machine learning. Le but étant de pouvoir prédire, une fois les modèles suffisamment entraînés, où les mesures ont été effectuées. Retirant ainsi la partie où les volontaires doivent manuellement renseigner des informations. Après cela, les données sont visualisées grâce à Grafana et stockées sur une base de données Hive (ainsi qu&#39;une base de données PostgreSQL). Il est ainsi possible de visualiser la qualité de l&#39;air à différents moments et différents endroits.

  

Notre projet vient s&#39;inscrire dans la partie machine learning et traitement des données. Précédemment à notre projet le traitement et le machine learning ne se font pas entièrement automatiquement. Notre but est d&#39;automatiser ces étapes en un pipeline, grâce aux microservices. Pour cela il nous à fallu comprendre la technologie Kafka qui est notre base pour les microservices, mais également comprendre le code existant. Il a fallu transformer le code existant pour qu&#39;il puisse être transformé en microservices. Nous avons dû organiser notre groupe en différentes équipes responsables de différentes parties du projet.

  

Dans ce rapport nous allons vous présenter la démarche que nous avons entrepris pour définir notre architecture, ensuite nous vous présenterons les détails de notre architecture et son implémentation.

  

La première étape dans notre démarche à été la toute première réunion de présentation, où le projet Polluscope ainsi que notre mission nous ont été présentés.



## Démarche du projet



## Phase de recherche

  

Suite à la réunion de présentation du projet, il nous a été donné pour tâche de nous renseigner sur le concept de microservices. Nous étions déjà familiers avec les notions de Service Oriented Architecture (SOA) il s&#39;agissait donc de voir les différences entre les deux concepts. Pour cela, il nous a été demandé de lire un article concernant l&#39;architecture Microservice. De cet article, nous avons pu retenir les principaux avantages et inconvénients des architectures à base de microservices.

  

Pour ce qui est des avantages, les microservices sont globalement plus indépendants, que ce soit d&#39;un point de vue du développement et du déploiement, c&#39;est-à-dire qu&#39;on peut déployer un microservice indépendamment des autres, permettant de développer chaque service également indépendamment. On retient également que chaque service est scalable en fonction des besoins. La quantité de code pour créer un microservice est minimale, ce qui permet de diviser le travail en de petites équipes de développement. Enfin les microservices satisfont les critères habituels de résilience puisqu&#39;en fonction de l&#39;architecture de l&#39;application, elle peut continuer à fonctionner même si un service est indisponible.

  

Pour ce qui est des inconvénients, l&#39;article note que même si la complexité de chaque microservice est relativement faible, la complexité de l&#39;application en elle-même est augmentée. Il faut donc bien planifier l&#39;architecture qu&#39;on va adopter. Comme autre inconvénient notable, l&#39;article présente l&#39;augmentation potentielle du trafic réseau dû à la communication entre les multiples microservices.

  

L&#39;article donne nous donne également des conseils sur les bonnes pratiques à adopter lorsqu&#39;on travaille avec des microservices.

  

En plus de ce premier article, nous avons eu un autre article destiné à nous familiariser avec Apache Kafka. Cet article d&#39;introduction nous explique les concepts de base de l&#39;architecture microservice avec Kafka, comment se servir de Kafka pour se connecter à une base de données, quels langages adopter, le concept de streaming et manière générale comment mettre en place une architecture avec Kafka. Cet article présente les concepts fondamentaux de Kafka, mais ne présente pas de cas pratique.

  

Après une réunion où nous avons expliqué ce qu&#39;on nous avions retenu du sujet et ce que nous avons compris des architectures microservice, nous sommes passés à l&#39;étape suivante du projet.


## Phase d&#39;apprentissage et découverte

  

Dans cette phase nous sommes passés à l&#39;apprentissage de Kafka, cette fois bien plus en profondeur. Par le biais de nombreux articles et tutoriels nous sommes devenus familiers avec les concepts de connecteurs (Kafka Connect), de pipeline, de streaming, de topic et du langage KSQL. Cette fois les articles étaient plus proches de tutoriels, certains d&#39;entre eux présentaient même des projets de machine learning ce qui se rapproche fortement de ce que nous devons réaliser ce projet. Grâce à ces tutoriels , nous avons commencé à imaginer la position qu&#39;allait prendre Kafka au sein de notre architecture. Suite à ces tutoriels, Hicham à entrepris de créer un [tutoriel](https://github.com/zekriHichem/Postgres_kafka_Python) pour tester les concepts dont nous pourrions avoir besoin pour ce projet. Ce tutoriel nous a plus tard servi de point de départ pour l&#39;implémentation du projet.

  

Après l&#39;apprentissage de Kafka nous avons commencé à découvrir le code existant, cette partie est probablement une des plus compliquées de tout le projet, la compréhension complète du code s&#39;est étendue jusqu&#39;à la complétion du projet. Au départ , nous avons simplement compris le fonctionnement général du code. Bien évidemment, avec la découverte du code, nous nous sommes posés de nombreuses questions. Pour répondre à ces questions , nous avons toujours pu compter sur Hafsa, que nous remercions. Enfin, au fur et à mesure des itérations de notre exploration, nous avons atteint un point où nous étions suffisamment confiants en notre compréhension pour commencer à planifier l&#39;architecture de notre pipeline.



## Phase de planification de l&#39;architecture

  

Après cette phase d&#39;apprentissage et de découverte, il a fallu appliquer nos nouvelles connaissances à notre contexte. Nous sommes passés par de nombreuses itérations d&#39;architectures. Dans un premier temps nous avons créé une architecture globale, c&#39;est à dire qu&#39;elle ne rentre pas dans les détails entre services mais présente l&#39;infrastructure globale du projet :

  

![](RackMultipart20220312-4-19pdh7a_html_8f83c9d217391591.png)

  

Schéma de l&#39;infrastructure générale de l&#39;application :

  

Ensuite nous avons dû créer un schéma du pipeline comme nous pensions le créer plus détaillée, nous avons obtenu le schéma de pipeline suivant :

  

![](RackMultipart20220312-4-19pdh7a_html_284880e7bee09d83.png)

  

Schéma du pipeline de l&#39;application :

  

On obtient deux branches de pipeline distincts avec 4 microservices, d&#39;un côté Preprocessing - TimeSeries et Classification - Validation, et de l&#39;autre Preprocessing - GPS et Postprocessing, on peut voir que Postprocessing a à la fois besoin des données de Classification - Validation et Preprocessing - GPS. Ensuite en entrée et sortie nous avons des connecteurs à PostgreSQL. Ce schéma nous a servi de point de départ pour l&#39;organisation de l&#39;étape suivante du projet, mais depuis il à eu quelques évolutions dans notre schéma.

  

Tout d&#39;abord une version avec une API qui serait proéminente :

  

![](RackMultipart20220312-4-19pdh7a_html_e617eaeb1d3b425e.png)

  

Schéma détaillé de l&#39;infrastructure avec API :

  

Cependant ce modèle précis d&#39;API n&#39;est pas adapté aux microservices et même contraire au principe même. Cependant ce schéma est plus détaillé et nous l&#39;avons donc repris dans la version finale :

  

![](RackMultipart20220312-4-19pdh7a_html_150d9cb75048a759.png)

  

Schéma détaillé de l&#39;infrastructure finale :

  

Il faut noter ici que nous avons 5 microservices, l&#39;équipe GPS à du rajouter le microservice **Stop&amp;Move Detection**.

## Mise en Place

  

Suite à la phase de planification est venue la phase d&#39;implémentation, pour cela nous avons séparé notre groupe en sous équipes, de la manière suivante (&quot;chefs d&#39;équipe&quot; en gras):

  

-  **Equipe Kafka/BD** en charge des connecteurs, de l&#39;infrastructure Kafka en générale et de la connexion à la base de données : **Mohamed Ramdane Debiane** , Mamadou Ndiaye, Julien Jacquet.

-  **Equipe Pipeline TimeSeries** qui regroupe les microservices Preprocessing - TimeSeries, Classification - Validation et Postprocessing : **Sidi Mohamed Hicham Zekri** ,Kamilia Loualia,Khaoula Hassoune, El Mamoun Affane-Aji .

-  **Equipe Pipeline GPS** qui regroupe les microservices Preprocessing - GPS , Stop&amp;Move Detection et Postprocessing : **Julien Daurat** , Amir Hammad, Abdellatif Missoumi et Séverine Selaquet.

  

Les deux équipes ont travaillé sur le service **Postprocessing**.

  

Nous avons également établi une liste de normes que nous devrions suivre pour l&#39;implémentation du code, pour ce qui est du code python nous avons décidé de suivre la norme PEP 8. Nous nous sommes également mis d&#39;accord pour suivre un certain nombre de règles, comme par exemple des noms de fonctions clairs et distincts et non personnels, ou bien encore faire en sorte que l&#39;adresse &quot;/&quot; de chaque microservice retourne &quot;hello, [nom du microservice]&quot;. Et enfin quelques règles élémentaires sur l&#39;éthique de travail à adopter.

  

Notre code est regroupé sur github sous une [organisation](https://github.com/uvsq-polluscope). Chaque équipe a évolué de manière relativement indépendante, comme le permet l&#39;idée de microservices. Sauf pour le service Postprocessing puisque les deux équipes ont travaillé dessus dans la continuité de leur branche de pipeline respectifs.



## Architecture du système

  

Dans cette partie, nous allons détailler chacun des composants que nous avons implémentés pour la mise en place du pipeline de traitement des données.

  

Il nous a été fourni par l&#39;université un espace serveur distant sur lequel reposera l&#39;ensemble de l&#39;infrastructure et des microservices. L&#39;hébergement permet un accès plus facile aux données et aux services par les doctorants et enseignants voulant travailler sur le sujet.

  

Pour garantir une portabilité des éléments et une intégration plus facile sur le serveur cible, nous avons utilisé les outils de conteneurisation que sont Docker et Docker-Compose. Ainsi, tous les composants uniques sont isolés et leur instanciation se fait à partir d&#39;un Dockerfile dédié. La composition de ces services se fait grâce à un fichier Docker-Compose, nous avons actuellement deux fichiers de composition de services, le premier instancie la plateforme Kafka ainsi que les bases de données PostgreSQL qui sont au nombre de deux (une stockant les données brutes et la seconde qui servira d&#39;entrepôt de données pour sauvegarder le résultat des traitements du pipeline).

Pour ce qui est du second fichier docker-compose, il sera exécuté après le lancement de l&#39;infrastructure et permettra de lancer ensemble les 5 micros services cités précédemment.

  

Nous avons tenu à séparer les deux implémentations (Infrastructure et Services) pour faciliter la maintenance des sous-projets et le travail collaboratif. Le détail des commandes permettant de mettre en service l&#39;un et les autres est contenu dans les fichiers README.md respectifs des sous-projets.

  

**Remarque:** Veillez à bien lancer les commandes du sous-projet **infrastructure** en premier lieu puis celles du projet **microservices.** Une démonstration détaillée du travail réalisé sera effectuée et permettra de clarifier les questions relatives aux différentes manipulations.

  

Nous allons maintenant passer à la description détaillée des choix d&#39;implémentations que nous avons effectués pour chacun des composants de l&#39;infrastructure.


## Détails de l&#39;infrastructure

  

Nous avons choisis comme base de départ de notre infrastructure une distribution Kafka réalisée et maintenue par le groupe [Confluent](https://www.confluent.io/fr-fr/what-is-apache-kafka/) à laquelle nous avons ajouté des outils tels qu&#39;une interface utilisateur pour administrer les différents aspects de Kafka et deux bases de données PostgreSQL

  

La liste des services est la suivante :

  

- Apache Zookeeper

- Apache Kafka

- Kafka Connect

- Schema-Registry

- Postgis

- Kafka-UI


### Apache Zookeeper

  

ZooKeeper est un service centralisé permettant de gérer les informations de configuration, de nommer, de fournir une synchronisation distribuée et de fournir des services de groupe.

  

Tous ces types de services sont utilisés sous une forme ou une autre par les applications distribuées. Chaque fois qu&#39;ils sont mis en œuvre, il y a beaucoup de travail qui doit être fait par le développeur d&#39;application et qui s&#39;avère très coûteux en temps et ressources en raison de la difficulté de mise en œuvre de ces types de services, ce qui les rend fragiles en présence de changement et difficiles à gérer. Même lorsqu&#39;elles sont effectuées correctement, différentes implémentations de ces services entraînent une complexité de gestion lorsque les applications sont déployées.

  

Nous avons donc utilisé Zookeeper pour garantir la fiabilité du cluster si jamais il est nécessaire de réaliser du scaling en ajoutant des brokers Kafka.

  

Pour notre démonstration, nous n&#39;avons déployé qu&#39;un seul et unique broker car celà était suffisant pour les besoins de notre pipeline.

  

La configuration initiale est la suivante :

  

![](RackMultipart20220312-4-19pdh7a_html_897084f44d29467b.png)

  

Nous avons ajouté un volume pour rendre les configurations de Zookeeper persistantes et nous avons exposé le port par défaut (ici : 2181).


### Apache Kafka

  

Une fois le gestionnaire de cluster mis en place,nous avons implémenté un broker Kafka, qui servira d&#39;intermédiaire entre les Producers/Consumers. Cet unique broker aura pour fonctionnalité principale de stocker les données arrivant dans les différents topics. Comme nous n&#39;avons ici qu&#39;une seule instance de broker, le facteur de réplication des topics est désactivé (aspect qui être corrigé par l&#39;ajout d&#39;un moins un broker supplémentaire). C&#39;est l&#39;adresse ip de ce broker sur notre réseaux qui sera fournie au microservices ce qui permettra ainsi le transfert de données entre les différentes instances.

  

Le service docker-compose est implémenté comme suit :

  

![](RackMultipart20220312-4-19pdh7a_html_9604072cb16f7e28.png)

  

Ici aussi, nous rendons les données contenues dans le broker persistantes, nous exposons deux ports 9092 (par défaut) et 9997 (Custom). Nous identifions cette instance de Kafka avec l&#39;id 1 et nous le relions à Zookeeper en re-définissant la variable d&#39;environnement KAFKA\_ZOOKEEPER\_CONNECT et on lui attribuant l&#39;adresse de l&#39;instance de zookeeper mentionnée plus haut.

  

L&#39;ajout d&#39;un nouveau broker peut être réalisé en ajoutant un service avec la même configuration. Il suffira seulement d&#39;incrémenter l&#39;identifiant de 1 et d&#39;exposer un autre port si nécessaire (pour éviter les ambiguïtés).



### Kafka Connect

  

Pour répondre à toutes les exigences fonctionnelles du projet, il nous a été nécessaire de mettre en place un système permettant l&#39;extraction des données traitées vers un entrepôt de données pour y faire des analyses ultérieures ainsi que de la visualisation. Telle est l&#39;utilité principale de l&#39;outil Kafka Connect permettant de diffuser des données de manière évolutive et fiable entre Apache Kafka® et d&#39;autres systèmes de données. Il simplifie la définition rapide de connecteurs qui déplacent de grands ensembles de données vers et depuis Kafka. Kafka Connect peut ingérer des bases de données entières ou collecter des métriques. Un connecteur d&#39;exportation (Sink Connector) peut fournir des données à partir de topics Kafka dans des index secondaires comme Elasticsearch ou dans des systèmes par lots tels que Hadoop pour une analyse hors ligne. Dans notre cas précis d&#39;implémentation, nous avons choisi d&#39;utiliser le connecteur [JDBC (Postgres)](https://docs.confluent.io/cloud/current/connectors/cc-postgresql-source.html) car c&#39;est un connecteur générique qui s&#39;adapte très bien au cas des bases de données relationnelles grâce au driver JDBC.

  

L&#39;implémentation de connect est réalisée de cette façon dans le fichier docker-compose :

  

![](RackMultipart20220312-4-19pdh7a_html_488f5656de103bfd.png)

  

Le service Kafka Connect expose une API REST qui permet via des requêtes HTTP de manipuler les connecteurs (création, mise à jour, suppression etc…). Une configuration de connecteur correspond à un fichier JSON qui est envoyé comme _ **payload** _ de la requête HTTP.

  

Voici un exemple de fichier de configuration pour un connecteur de type SINK vers une base de donnée PostgreSQL :

  

![](RackMultipart20220312-4-19pdh7a_html_554a33b1203104fb.png)

  

Les paramètres importants ici sont **connector.class** et **connection.url** qui décrivent le type de connecteur utilisé et à quelle base il doit être relié. Les paramètres **key/value.converter** définissent le type de données qui est manipulé par les connecteurs (ici nous manipulons la clé du topic comme étant une chaîne de caractère et les données comme étant de type Avro (voir section suivante). Nous relions ce connecteur à notre instance de Schema-Registry (voir section suivante) et nous définissons quelques paramètres concernant les topics et les schémas cibles:

  

- auto-create/auto-evolve: permet de créer la table si elle n&#39;existe pas dans la base de donnée cible et permet de faire une mise à jour de la table si le schéma contenu dans Registry évolue au cours du temps.

- topics: Représente la liste des topics qu&#39;il faut externaliser.

- mode: Défini quelle partie du topic il faut externaliser (ici la valeur et non la clé)

- field: Défini quelle partie du topic fera office de clé primaire dans la table relationnelle.

- mode: Défini le monde d&#39;insertion des données de topic, ici upsert signifie que le tuple est mis à jour s&#39;il existe déjà et inséré sinon.

  

Nous avons automatisé la création des connecteur avec l&#39;aide d&#39;un seul script bash qui attends que Kafka Connect et le Broker soit actif, puis va lire tous les fichiers de configuration et les envoyer avec la commande curl

  

![](RackMultipart20220312-4-19pdh7a_html_c320156c745ed282.png)



### Schema-Registry

  

Confluent Schema Registry fournit une couche de service pour vos métadonnées. Il fournit une interface RESTful pour stocker et récupérer vos schémas Avro®, JSON Schema et Protobuf. Il stocke un historique versionné de tous les schémas en fonction du nom de topic spécifié et offre une compatibilité multiple. Il fournit des sérialiseurs qui se connectent aux clients Apache Kafka® qui gèrent le stockage et la récupération de schéma pour les messages Kafka qui sont envoyés dans l&#39;un des formats pris en charge. Schema Registry vit en dehors et séparément des Broker Kafka.Les producers/consumers continuent de communiquer avec Kafka pour publier et lire des données (messages) dans des topics. Parallèlement, ils peuvent également communiquer avec Schema Registry pour envoyer et récupérer des schémas décrivant les modèles de données. pour les messages.

  

![](RackMultipart20220312-4-19pdh7a_html_f75f2e657fa092bd.png)

  

Parmi tous les formats de données pris en charge par Registry, nous avons choisi Avro car il permet un typage strict des données, la possibilité de définir des valeurs par défaut etc… Le schéma de données ainsi défini doit être ajouté par chaque Producer à la création et l&#39;envoi d&#39;un message.

  

![](RackMultipart20220312-4-19pdh7a_html_167780d792aa018e.png)

  

1.

### Kafka-UI

  

Pour simplifier l&#39;administration de tous les éléments relatifs à Kafka, nous avons ajouté à notre infrastructure, une interface web accessible depuis [http://192.168.33.124:9000/ui](http://192.168.33.124:9000/ui).

  

Cette interface regroupe en outre, des informations sur le cluster et zookeeper, comme le nombre de brokers online, le nombre de topics et leur facteur de réplication et la quantité de données produites et consommées.

  

Elle permet aussi la création de topics, de schémas et de connecteurs de façon intuitive et graphique

  

![](RackMultipart20220312-4-19pdh7a_html_cc253edcaf5e8fe0.png)

  

![](RackMultipart20220312-4-19pdh7a_html_e6bbef9478dede2b.png) ![](RackMultipart20220312-4-19pdh7a_html_a7a9fee4cf749944.png)



### Bases de données PostgreSQL

  

Pour finir, nous avons dans notre fichier de composition de services deux bases de données relationnelles, qui se placent aux extrémités de notre pipeline. Pour servir d&#39;espace de stockage pour les données brutes et pour les données traitées. Ces bases sont reliées à des connecteurs Source/Sink.

  

Ces bases de données sont initialisées avec un schéma et des données initiales à la création de leur conteneur respectif et donc aucune autre configuration n&#39;est nécessaire. Elle sont directement accessible par un client depuis les ports exposés dans les configurations docker-compose suivantes :

  

![](RackMultipart20220312-4-19pdh7a_html_4b267dfbe2bb41fc.png) ![](RackMultipart20220312-4-19pdh7a_html_64910ee26aa2b682.png)

  

Exemple : $\&gt; psql -U postgres -h 192.168.33.124 -p 5435 #accède la base rawdata



## Détails pour la partie Timeseries



## Détails pour la partie GPS

  

Pour l&#39;équipe GPS, deux microservices ont été créés, il s&#39;agit de **&quot;preprocess gps&quot;** et de **&quot;stop move detection&quot;**. Chacun d&#39;entre eux a une tâche principale à accomplir. Le premier est chargé de recevoir les données des participants, puis de les traiter et de les renvoyer à la base de données. Le second doit, quant à lui, détecter l&#39;arrêt de mouvement du participant. Les deux microservices communiquent avec la base de données Postgres et la plateforme Kafka.

  

1.

### Preprocess GPS :

  

Le microservice **Preprocess GPS** communique avec la plateforme Kafka en lisant les données du topic **rawdataGPS** , puis il les traite et les renvoie à la plateforme Kafka en les écrivant sur le topic **ProducerRawDataGPS**.

  

![](RackMultipart20220312-4-19pdh7a_html_450e1987b5d7f125.jpg)

  

Afin que notre microservice puisse communiquer avec Kafka, nous devons définir un ensemble de paramètres (ID de groupe, comment sérialiser et désérialiser les données, ...etc.). Ces derniers sont présentés dans les figures ci-dessous :

  

![](RackMultipart20220312-4-19pdh7a_html_74ea003e127a9c56.png)

  

![](RackMultipart20220312-4-19pdh7a_html_4224cc8d8d9b9f7d.png)

  

Ainsi Preprocess GPS permet de stocker les messages reçus dans un dictionnaire python. A chaque fois qu&#39;il reçoit un message il vérifie si la clé existe dans le dictionnaire, si ce n&#39;est pas le cas il y ajoute la clé avec la valeur correspondante. Il est également important de s&#39;assurer que les messages d&#39;un utilisateur ne dépassent pas un certain nombre.En ce sens, la notion de semaine à était établie. A chaque début d&#39;une nouvelle semaine, nous appelons la fonction **&quot;data\_pre\_processing\_gps&quot;** , qui fait ce qui suit :

  

- Rastériser les données.

- Obtenir l&#39;identifiant maximal du message à partir de la trame de données.

- Créer un index de Hilbert pour chaque point.

- Stocker les données dans la table &quot;  **clean\_gps&quot;**.

- Créer le dataframe **clean\_gps\_with\_activities** qui sera utilisé par la suite.

  

![](RackMultipart20220312-4-19pdh7a_html_5088d49dd611e598.png)

  

Une fois les données traitées, le microservice les renvoie à la plateforme Kafka en écrivant dans le topic ProducerRawDataGPS avec le schéma suivant :

  

-  **id :** int

-  **table\_id :** int

-  **lat :** float

-  **lon :** float

-  **timestamp :** string

  


### Stop move detection :

  

## Conclusion

  

Projet complèté + … + 11 personnes pas toujours facile de s&#39;organiser

  

## Bibliographie