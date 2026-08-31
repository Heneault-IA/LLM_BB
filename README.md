# LLM_BB — Assistant de règles Blood Bowl (RAG local & multimodal)

Un assistant qui répond à des questions de règles de Blood Bowl en s'appuyant uniquement sur le PDF officiel du règlement, via un pipeline RAG (Retrieval-Augmented Generation) 100% local.

Projet personnel, inspiré des vidéos de [Harish Neel](https://www.youtube.com/@harishneel1).

## Pourquoi ce projet

Blood Bowl a un règlement dense, plein de cas particuliers et d'exceptions. L'objectif : pouvoir poser une question en langage naturel ("Dois-je relancer un jet de blocage en cas de frénésie ?") et obtenir une réponse fondée sur le texte réel des règles — pas une réponse inventée par le modèle.

## Fonctionnement

```
PDF des règles
      │
      ▼
Extraction (unstructured, hi_res)
  texte · tableaux (HTML) · images (base64)
      │
      ▼
Découpage par titres / sections
      │
      ▼
Résumé enrichi par IA pour les chunks avec table ou image
  (le VLM décrit le visuel → ce résumé devient le contenu embeddé,
   le texte/tableaux/images bruts sont conservés à part)
      │
      ▼
Indexation vectorielle (Chroma + embeddings multilingues)
      │
      ▼
Question utilisateur → retrieval top-k → reconstruction du contexte
(texte + tableaux + images d'origine)
      │
      ▼
Génération de la réponse par un LLM local (Ollama)
  → bascule automatique vers un modèle vision si le contexte contient une image
```

## Stack technique

- **Extraction PDF** : [`unstructured`](https://github.com/Unstructured-IO/unstructured) (stratégie `hi_res`, extraction de tables et d'images)
- **Découpage** : chunking par titre (`chunk_by_title`)
- **Embeddings** : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (via `langchain-huggingface`)
- **Base vectorielle** : Chroma
- **Génération** : [Ollama](https://ollama.com/), en local
  - `qwen3:1.7b` pour les réponses textuelles
  - `qwen2.5vl:3b` pour les chunks contenant des images
- **Orchestration** : LangChain

## Installation

Le projet utilise [`uv`](https://docs.astral.sh/uv/) pour la gestion des dépendances.

```bash
git clone <url-du-repo>
cd LLM_BB
uv sync
```

Il faut également [installer Ollama](https://ollama.com/download) et récupérer les modèles utilisés :

```bash
ollama pull qwen3:1.7b
ollama pull qwen2.5vl:3b
```

## Données

Le dossier `data/` (contenant le PDF du règlement) n'est **pas inclus** dans ce repo : il s'agit d'un document officiel soumis aux droits de Games Workshop.

Pour faire tourner le projet :
1. Procure-toi le règlement officiel de Blood Bowl (édition PDF)
2. Place-le dans `data/rules_bb.pdf`

## Limites connues

Les tests montrent que le retrieval colle encore assez peu à certaines questions, principalement parce que le domaine est très spécifique : un modèle d'embeddings généraliste représente mal le vocabulaire propre au jeu (noms de compétences, résultats de dés, mécaniques spéciales).

Pistes d'amélioration envisagées :
- vérifier le chunking et ce que remonte réellement le retrieval sur les questions ratées
- recherche hybride (vectorielle + mots-clés)
- tester un modèle d'embeddings plus performant, avant d'envisager un réentraînement sur le domaine