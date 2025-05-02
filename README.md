# 🐦 Analyseur de Sentiments de Tweets en Français

Cette application innovante utilise l'intelligence artificielle pour analyser automatiquement le ton émotionnel des tweets en français. Grâce à des modèles de traitement du langage avancés, elle détecte si un message exprime une opinion positive 😊, neutre 😐 ou négative 😠, offrant ainsi une compréhension immédiate des sentiments exprimés.

Le système fonctionne en deux étapes claires : il traduit d'abord le tweet en anglais à l'aide du modèle Helsinki-NLP, puis évalue son contenu émotionnel avec DistilBERT, un algorithme spécialisé dans l'analyse de sentiments. Cette double approche garantit des résultats précis malgré la barrière linguistique initiale.

L'outil propose deux modes d'utilisation flexibles : une analyse manuelle pour examiner des tweets individuels, et un traitement par lots permettant d'évaluer des centaines de messages simultanément via des fichiers CSV. Cette polyvalence le rend adapté aussi bien aux curiosités personnelles qu'aux études de marché approfondies.

Les résultats s'affichent dans une interface claire et interactive, mettant en valeur les tendances émotionnelles à travers des graphiques dynamiques et des statistiques détaillées. Les données peuvent être facilement exportées pour un archivage ou une analyse complémentaire dans d'autres logiciels.

Particulièrement accessible, cette solution peut être déployée en quelques minutes aussi bien sur un ordinateur personnel (via Streamlit) que sur le cloud (via Streamlit Cloud), ce qui en fait un outil immédiatement opérationnel pour les particuliers comme pour les professionnels des réseaux sociaux.
## 🌟 Fonctionnalités

- 🔍 **Analyse en temps réel** des tweets en français
- 🌐 **Traduction automatique** vers l'anglais
- 📈 **Visualisations interactives** avec Plotly
- 📁 **Support CSV** pour l'analyse par lots
- 🎨 **Interface intuitive** avec Streamlit

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.8+
- Git
- Compte Streamlit (pour le déploiement)

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/assietoundiaye/Analyseur_sentiment.git
cd Analyseur_sentiment

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
