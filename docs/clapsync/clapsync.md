# Manuel Utilisateur — clapsync

## 1. Introduction

**clapsync** est un outil de synchronisation et de découpe multi-caméras. Son objectif est d'aligner automatiquement plusieurs enregistrements vidéo tournés simultanément par des caméras différentes, puis d'en exporter des versions découpées et synchronisées.

### Points clés :

*   **Synchronisation automatique** via l'empreinte audio (MFCC).
*   **Édition manuelle** pour un ajustement fin.
*   **Export groupé** avec recadrage temporel identique pour tous les fichiers.

## 2. Préparation et Importation

### Prérequis
Avant de commencer, assurez-vous que :

1. Vous avez au moins **deux fichiers vidéo** (formats supportés : `.mp4`, `.mov`, `.avi`, `.mkv`, etc.).
2. Chaque vidéo dispose d'une **piste audio exploitable**.

### Sélection des fichiers
Au lancement, l'interface de sélection vous permet de constituer votre groupe de travail.

![Capture d'écran de l'interface de sélection : boutons Add Videos, Remove et liste des fichiers](select_videos_1.png)

*   **Ajouter des vidéos** : Cliquez sur **Add Videos...**. Vous pouvez importer plusieurs fichiers simultanément.
*   **Définir la référence** : L'ordre de la liste est crucial. La **première vidéo** sert de référence temporelle (décalage = 0). Utilisez les boutons **Move Up** / **Move Down** pour l'ajuster.
*   **Supprimer** : Retirez un fichier erroné avec le bouton **Remove**.
*   **Valider** : Cliquez sur **OK** (le bouton s'active dès que 2 vidéos sont présentes).

## 3. Synchronisation Automatique

Une fois la sélection validée, le logiciel lance l'analyse acoustique de manière autonome.

![Fenêtre de progression de l'analyse audio indiquant l'extraction et la corrélation](compute_offsets.png)

Le logiciel extrait les pistes audio et utilise des coefficients MFCC (*Mel-Frequency Cepstral Coefficients*) pour comparer les spectres sonores. Cette méthode est robuste, même si les microphones ont des sensibilités ou des égalisations différentes.

*   **Progression** : Une barre d'état indique l'étape en cours pour chaque fichier.
*   **Échec de détection** : Si le logiciel ne trouve pas de correspondance (audio trop différent), le décalage est mis à zéro. Vous pourrez le corriger manuellement à l'étape suivante.

## 4. L'Éditeur de Synchronisation

L'interface principale se divise en trois zones interactives :

![Vue d'ensemble de l'éditeur : grille de prévisualisation en haut et timeline en bas](editor_overview.png)

### 4.1 La Zone de Prévisualisation
Elle affiche une grille contenant tous vos flux vidéo synchronisés en temps réel.

*   **Lecture synchrone** : Lorsque vous lancez la lecture, toutes les caméras avancent ensemble selon leurs décalages respectifs. Vous visualisez instantanément si le calage est correct.

### 4.2 Les Contrôles de Lecture
Situés sous la grille, ils permettent de naviguer dans le temps :

*   **Play / Pause** : Bouton central ou touche `Espace`.
*   **Indicateur de temps** : Affiche le format `Position actuelle / Durée totale`.
*   **Mode Loop** : Cochez cette case pour répéter la zone de découpe sélectionnée en boucle.

### 4.3 La Timeline (Ajustements manuels)
La timeline permet de visualiser et d'ajuster le calage de chaque piste sur l'axe temporel global.

![Détail de la timeline : pistes colorées, icône de cadenas sur la piste de référence et poignées de découpe](timeline_details.png)

#### Navigation et Zoom

*   **Déplacer la lecture** : Cliquez n'importe où sur la règle temporelle ou faites glisser la **tête de lecture** (trait rouge vertical).
*   **Zoomer** : `Ctrl + Molette` pour agrandir ou réduire l'échelle du temps.
*   **Défilement** : Utilisez la `Molette` simple pour naviguer horizontalement.

#### Correction du décalage (Offset)
Si le calcul automatique n'est pas parfait :

1. **Double-cliquez** sur le bloc coloré de la vidéo à corriger (le bloc gris, de référence, n'est pas modifiable).
2. Saisissez la valeur exacte en secondes dans la fenêtre **Set Offset**.
3. **Astuce** : Repérez un événement visuel et sonore bref (un clap) et alignez les pistes visuellement sur ce point précis.

#### Définition de la zone de découpe
Les **poignées grises** aux extrémités de la timeline définissent l'intervalle de temps qui sera conservé à l'export.

*   **Début** : Faites glisser la poignée gauche vers la droite.
*   **Fin** : Faites glisser la poignée droite vers la gauche.
*   Les zones grisées à l'extérieur des poignées ne seront pas exportées.

## 5. Exportation des Résultats

Une fois le réglage terminé, cliquez sur le bouton **Export**.

![Fenêtre des paramètres d'export : menus de résolution, FPS et choix du dossier](export_settings.png)

### Configuration de la sortie

*   **Résolution** : Choisissez la résolution de sortie absolue dans la liste déroulante. Les options sont calculées à partir de la plus petite résolution détectée parmi vos fichiers (facteurs 1×, 0.75×, 0.5×, 0.25×). Toutes les vidéos sont redimensionnées vers cette résolution commune, quelle que soit leur résolution d'origine.
*   **Fréquence (FPS)** : Par défaut réglé sur la fréquence minimale détectée. Valeurs possibles : 1 à 240 FPS.
*   **Destination** : Cliquez sur **Browse...** pour choisir le dossier d'enregistrement.

### Processus final
Le logiciel génère pour chaque caméra un nouveau fichier nommé `{nom_original}_synced.mp4`. 

> **Note technique** : Si une vidéo commence après le point de début de la découpe, le logiciel insère automatiquement des images noires au début du fichier pour garantir que toutes les vidéos exportées soient parfaitement alignées sur la même "frame 0".

## 6. Résultat attendu
À la fin du processus, vous obtenez un dossier contenant $N$ fichiers vidéo :

1. **Synchronisés** : Ils commencent tous au même instant temporel.
2. **Identiques en durée** : Parfaits pour une importation directe dans un pipeline de reconstruction en aval.