# Manuel Utilisateur — clapsync

**clapsync** synchronise et découpe des enregistrements multi-caméras à partir
de leur piste audio, puis en exporte des copies alignées et de durée identique.

Ce manuel décrit l'application graphique (GUI). Pour l'usage en ligne de commande
ou via l'API Python, voir le [README](../../README.md) du projet.

---

## Sommaire

1. [Introduction](#1-introduction)
2. [Installation et lancement](#2-installation-et-lancement)
3. [Sélection des fichiers](#3-sélection-des-fichiers)
4. [Synchronisation automatique](#4-synchronisation-automatique)
5. [L'éditeur de synchronisation](#5-léditeur-de-synchronisation)
6. [Exportation des résultats](#6-exportation-des-résultats)
7. [Résultat attendu](#7-résultat-attendu)
8. [Dépannage](#8-dépannage)
9. [Annexe — Raccourcis clavier](#9-annexe--raccourcis-clavier)

---

## 1. Introduction

Filmez un même instant avec plusieurs caméras ou enregistreurs démarrés à des
moments différents, puis laissez clapsync les aligner en écoutant leur audio.
L'outil produit des copies découpées qui commencent toutes au même instant et
ont exactement la même durée — prêtes pour un montage multicam ou un pipeline de
reconstruction en aval.

Idéal pour les tournages multicam, interviews, concerts, et toute prise devant
laquelle vous avez claqué un clap. 👏

### Points clés

* **Synchronisation automatique** par empreinte audio (coefficients MFCC),
  robuste aux différences de micros et de gain.
* **Précision inférieure à l'image** : les décalages sont conservés en secondes
  flottantes, donc audio et vidéo restent calés même quand le décalage réel
  tombe entre deux images.
* **Édition manuelle** pour corriger un calage ou définir la zone de découpe.
* **Export groupé** vers une résolution et une fréquence communes.

---

## 2. Installation et lancement

### Option A — Installateur Windows

Téléchargez `clapsync-setup-<version>.exe` et exécutez-le. L'installation se fait
par utilisateur dans `%LOCALAPPDATA%\clapsync` puis télécharge l'environnement
verrouillé (~2 Go de téléchargement, ~8 Go sur disque) : **une connexion internet
est requise pendant l'installation**.

> L'installateur n'est pas signé : au premier lancement, Windows SmartScreen
> affiche un avertissement « application non reconnue ». Choisissez
> **Informations complémentaires → Exécuter quand même**.

Aucun GPU NVIDIA n'est nécessaire pour installer. Sans GPU, clapsync fonctionne
sur le processeur (synchronisation et export plus lents).

### Option B — Depuis les sources ([pixi](https://pixi.sh))

```bash
pixi install
pixi run clapsync
```

Au lancement, la fenêtre de sélection des fichiers s'ouvre.

---

## 3. Sélection des fichiers

Constituez ici votre groupe de vidéos à synchroniser.

![Interface de sélection : liste des fichiers et boutons Add Videos, Remove, Move Up, Move Down](select_videos_1.png)

* **Ajouter des vidéos** : cliquez sur **Add Videos…** et choisissez un ou
  plusieurs fichiers. Formats acceptés : `.mp4`, `.mov`, `.avi`, `.mkv`,
  `.mts`, `.m2ts`, `.webm`, `.flv`, `.wmv`.
* **Définir la référence** : l'ordre de la liste compte. La **première vidéo**
  sert de référence temporelle (décalage = 0) ; toutes les autres sont calées
  par rapport à elle. Utilisez **Move Up** / **Move Down** pour réordonner.
* **Supprimer** : sélectionnez un ou plusieurs fichiers et cliquez sur
  **Remove**.
* **Valider** : cliquez sur **OK**. Le bouton s'active dès que **2 fichiers au
  moins** sont présents.

![Liste peuplée de plusieurs fichiers vidéo prête à être validée](select_videos_2.png)

> **Prérequis** : chaque fichier doit posséder une piste audio exploitable —
> c'est elle que la synchronisation écoute.

---

## 4. Synchronisation automatique

Après validation, clapsync analyse l'audio de chaque fichier sans intervention.

![Fenêtre de progression de l'analyse audio](compute_offsets.png)

Les pistes audio sont extraites puis comparées par corrélation de leurs
coefficients MFCC (*Mel-Frequency Cepstral Coefficients*), une empreinte
robuste aux différences de micros et d'égalisation. Un pic de corrélation donne
le décalage de chaque fichier, affiné en dessous de l'image.

* **Progression** : une barre indique l'avancement. L'opération peut être
  annulée.
* **Échec de détection** : si aucune correspondance nette n'est trouvée (audio
  trop différent, piste silencieuse), le décalage concerné est laissé à zéro.
  Vous le corrigerez à la main dans l'éditeur (voir §5).

---

## 5. L'éditeur de synchronisation

> **Proxies de prévisualisation (sources haute résolution).** Si l'un de vos
> fichiers dépasse 1080p, clapsync propose de générer des *proxies* légers en
> 480p pour une lecture fluide dans l'éditeur. Répondez **Yes** (recommandé, par
> défaut) pour lancer un transcodage unique, ou **No** pour lire les originaux.
> L'export utilise toujours les fichiers d'origine en pleine résolution, quel
> que soit ce choix.

L'éditeur regroupe trois zones : la prévisualisation, les contrôles de lecture
et la timeline.

![Vue d'ensemble de l'éditeur : mosaïque de prévisualisation en haut, timeline en bas](editor_overview.png)

### 5.1 La prévisualisation

Une mosaïque affiche tous les flux en parallèle, chacun surmonté du nom de son
fichier. Les caméras avancent ensemble selon leurs décalages : vous voyez
immédiatement si le calage est correct. Un voile **« Loading… »** apparaît
brièvement lors des chargements ou des sauts importants.

### 5.2 Les contrôles de lecture

Situés entre la mosaïque et la timeline :

* **▶ Play / ⏸ Pause** : bouton de gauche, ou touche `Espace`.
* **Indicateur de temps** : format `position / durée` (`m:ss.mmm`), calé sur la
  zone de découpe.
* **Loop** : coché par défaut. La lecture reboucle en continu sur la zone de
  découpe. Décochez pour qu'elle s'arrête à la fin de la zone.

### 5.3 La timeline

La timeline place chaque piste sur l'axe temporel global. La piste de référence
est **grise et verrouillée** (icône cadenas) ; les autres sont colorées.

![Détail de la timeline : pistes colorées, cadenas sur la référence, poignées de découpe](timeline_details.png)

#### Navigation et zoom

* **Déplacer la lecture** : cliquez ou faites glisser n'importe où pour amener
  la **tête de lecture** (trait rouge) à cet instant.
* **Zoomer** : `Ctrl + Molette` agrandit ou réduit l'échelle de temps autour du
  curseur.
* **Défiler** : la `Molette` seule fait défiler horizontalement. Une barre
  verticale apparaît si les pistes ne tiennent pas toutes en hauteur.

#### Corriger un décalage (offset)

Si le calcul automatique n'est pas parfait :

1. **Double-cliquez** sur le bloc coloré de la piste à corriger (le bloc gris de
   référence n'est pas modifiable).
2. Dans la fenêtre **Set Offset**, saisissez le décalage exact en secondes
   (précision au millième).
3. **Astuce** : repérez un événement bref à la fois visible et sonore (un clap)
   et calez les pistes visuellement sur ce point.

Ajuster un décalage réaligne aussitôt la prévisualisation et peut resserrer la
zone de découpe si nécessaire.

#### Définir la zone de découpe

Les **poignées** aux deux extrémités délimitent l'intervalle conservé à
l'export. À l'ouverture, la zone est réglée sur le **chevauchement commun** à
toutes les pistes (la portion où toutes sont présentes).

* Glissez la poignée **gauche** vers la droite pour retarder le début.
* Glissez la poignée **droite** vers la gauche pour avancer la fin.
* Les zones grisées hors des poignées ne seront pas exportées.

---

## 6. Exportation des résultats

Réglage terminé, cliquez sur **Export…**.

![Fenêtre des paramètres d'export : résolution, fréquence et dossier de sortie](export_settings.png)

Un résumé rappelle le nombre de vidéos, la zone de découpe et sa durée. Réglez
ensuite :

* **Résolution** : menu déroulant. La première option, **Native**, correspond à
  la plus petite résolution détectée parmi vos fichiers ; les options suivantes
  proposent des hauteurs standard *inférieures* (2160p, 1440p, 1080p, 720p, 480p)
  au ratio d'origine. Toutes les vidéos sont redimensionnées vers cette
  résolution commune. clapsync n'agrandit jamais au-delà de la résolution
  source.
* **Fréquence (FPS)** : menu déroulant. **Native** correspond à la fréquence la
  plus basse détectée ; les options suivantes proposent des fréquences standard
  *inférieures* (60, 50, 30, 25, 24 fps).
* **Dossier de sortie** : saisissez un chemin ou cliquez sur **Browse…**.

Cliquez sur **OK** pour lancer l'export. Une fenêtre de progression s'affiche,
avec un bouton **Cancel** pour l'interrompre.

Chaque caméra produit un fichier `{nom_original}_synced.mp4` (les pistes
purement audio produisent un fichier audio `{nom_original}_synced.<ext>`).

> **Note technique — remplissage des trous.** Si une piste commence après le
> début de la zone de découpe, clapsync insère automatiquement des images noires
> (et du silence) au début du fichier ; de même en fin de piste. Toutes les
> vidéos exportées partagent ainsi la même « image 0 » et la même durée.

> **GPU vs CPU.** L'encodage utilise le GPU (NVENC `h264_nvenc`) quand il est
> disponible, avec bascule automatique sur le processeur (`libx264`) sinon.
> L'export CPU est plus lent mais donne le même résultat.

À la fin, un message récapitule les fichiers écrits (ou les erreurs éventuelles,
piste par piste).

---

## 7. Résultat attendu

Vous obtenez dans le dossier de sortie **N fichiers** :

1. **Synchronisés** : ils commencent tous au même instant.
2. **De durée identique** : prêts pour une importation directe dans un montage
   multicam ou un pipeline de reconstruction.

---

## 8. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| Le bouton **OK** de la sélection reste grisé | Moins de 2 fichiers dans la liste | Ajoutez au moins deux vidéos. |
| Un fichier est refusé à l'ouverture | Format ou fichier illisible | Vérifiez qu'il figure parmi les formats acceptés (§3) et qu'il n'est pas corrompu. |
| Une piste reste calée à 0 après l'analyse | Aucune correspondance audio nette (piste silencieuse, micro trop différent) | Corrigez le décalage à la main dans l'éditeur (§5.3). |
| Le calage est proche mais pas parfait | Précision limitée par l'audio | Zoomez sur un clap et affinez l'offset au millième (§5.3). |
| La prévisualisation reste noire / « Loading… » persiste | Décodage lent d'une source haute résolution | Patientez ; sur de très grosses sources, l'affichage se met à jour après le décodage. |
| L'export est lent | Encodage sur processeur (pas de GPU NVENC exploitable) | Normal sans GPU compatible. Une résolution ou une fréquence plus basse accélère l'export. |
| Certaines vidéos exportées commencent par du noir | Remplissage volontaire des trous | Comportement attendu : garantit une « image 0 » commune (voir §6). |
| SmartScreen bloque l'installateur | Installateur non signé | **Informations complémentaires → Exécuter quand même** (§2). |

---

## 9. Annexe — Raccourcis clavier

| Action | Raccourci |
|---|---|
| Lecture / Pause | `Espace` |
| Zoomer / dézoomer la timeline | `Ctrl + Molette` |
| Défiler la timeline | `Molette` |
| Déplacer la tête de lecture | Clic / glisser sur la timeline |
| Modifier le décalage d'une piste | Double-clic sur son bloc coloré |
