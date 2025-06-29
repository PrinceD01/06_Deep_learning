# Objectifs du projet

## Objectif principal

Analyser les réponses des citoyens dans le cadre de la consultation "Comment agir pour un tourisme plus durable en France ?", organisée par l’Office du Tourisme d’Ille-et-Vilaine, afin d’identifier les actions prioritaires simples à mettre en place pour favoriser un tourisme responsable.

## Intérêt pédagogique

* Appliquer une démarche complète de data analyse et de data storytelling sur un sujet d’intérêt sociétal.
* Travailler la transformation d’un grand volume de réponses textuelles en recommandations claires et concrètes.
* Développer des compétences de visualisation impactante et de synthèse orientée vers l’aide à la décision publique.

# Structure du projet

```
project/
│
├── data/                 # Dossier destiné à accueillir les données brutes et/ou nettoyées
│
├── notebooks/            # Dossier destiné à contenir les notebooks d'analyse
│
├── src/                  # Dossier pour des scripts de traitement ou d’analyse (actuellement vide)
│
├── outputs/              # Dossier de sortie avec les livrables finaux
│   ├── Consultation_Tourisme_Notebook.ipynb   # Notebook d'analyse et de traitement
│   ├── Presentation_of_results.pptx           # Support de présentation
│   └── Data_files/                             # Données nettoyées ou formatées si nécessaire
```

# Jeu de données

**Source :** Données issues d'une consultation citoyenne menée auprès de 1 500 répondants.

**Thématique :**
Tourisme durable en France.

**Contenu :**

* Question ouverte : *"Quelles actions simples pourrait-on mener en priorité pour tendre vers un tourisme responsable ?"*

**Format des données :**

* Données textuelles issues des réponses libres des participants.

# Méthodologie

1. **Préparation des données :**

   * Nettoyage des réponses textuelles : suppression des doublons, gestion des réponses vides ou non exploitables.
   * Normalisation et prétraitement linguistique : suppression des stop words, lemmatisation, traitement des fautes courantes.

2. **Analyse exploratoire :**

   * Identification des thématiques récurrentes via analyse de fréquence et nuages de mots.
   * Exploration des cooccurrences et des associations d’idées.

3. **Catégorisation des réponses :**

   * Regroupement des propositions similaires par familles d’actions (mobilité, hébergement, consommation locale, sensibilisation, etc.).

4. **Synthèse des actions prioritaires :**

   * Hiérarchisation des actions simples et réalisables, en fonction de leur fréquence d’apparition et de leur faisabilité.

5. **Restitution des résultats :**

   * Construction d’un support visuel et synthétique adapté à des décideurs publics et institutionnels.
   * Mise en avant des recommandations citoyennes sous forme de priorisation simple et actionable.
