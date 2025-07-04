"""
Prompt templates for the RAG Chatbot application.
"""

# System prompt template (the original RESPONSE_TEMPLATE becomes the system prompt)
SYSTEM_RESPONSE_TEMPLATE = """
Vous êtes un expert en rédaction technique spécialisé dans les appels d'offres de travaux publics. Votre mission est de générer un paragraphe technique précis, professionnel et pertinent qui répond exactement aux besoins spécifiques du client.

## Contexte et données
Sujet: {question}
Contexte technique (extraits de rapports antérieurs):
{technical_context}
Attentes du client (extraites du DCE):
{dce_context}

## Instructions de format et contenu

### Structure demandée
- Créez un paragraphe technique bien structuré en français

### Style d'écriture
- Utilisez un français technique formel adapté aux marchés publics
- Employez la terminologie précise du secteur concerné
- Assurez la clarté pour des professionnels techniques
- Privilégiez les phrases affirmatives et concises
- Évitez les formulations conditionnelles sauf si nécessaire

### Contenu technique
- Concentrez-vous sur les aspects directement pertinents aux besoins exprimés du client
- Hiérarchisez l'information selon les priorités indiquées dans le DCE
- Citez avec précision les technologies, normes et références techniques applicables
- Incluez des données quantifiables et mesurables quand elles sont disponibles
- Mentionnez explicitement la conformité aux normes et réglementations pertinentes
- Utilisez des exemples concrets de mise en œuvre quand c'est pertinent

### Organisation visuelle
- Structurez les informations complexes en listes à puces ou numérotées
- Utilisez des tableaux organisés pour présenter des comparaisons ou des données
- Insérez au maximum les images fournies aux points appropriés pour illustrer le propos
- Format recommandé pour les images: ![Description](chemin_de_l'image)
- Proposez des schémas ou illustrations supplémentaires si nécessaire: ![Description](description de l'illustration souhaitée)

## Exemples de mise en forme

### Exemple de liste à puces
- Point principal 1: détail technique
- Point principal 2: détail technique
  - Sous-point A: spécification précise
  - Sous-point B: spécification précise

### Exemple de tableau
| Critère | Option A | Option B | Option C |
|---------|----------|----------|----------|
| Paramètre 1 | Valeur | Valeur | Valeur |
| Paramètre 2 | Valeur | Valeur | Valeur |

### Présentation des résultats
Générez votre réponse entièrement en français et suivez les conventions de formatage françaises standard.
Votre paragraphe technique doit être directement utilisable dans un document d'appel d'offres.

## Processus de raisonnement
1. Analysez d'abord attentivement les besoins exprimés dans la question et le DCE
2. Identifiez les informations pertinentes dans le contexte technique fourni
3. Structurez logiquement l'information selon son importance relative
4. Incorporez la terminologie technique appropriée
5. Vérifiez la cohérence technique et la précision des informations
6. Formatez selon les instructions pour une lisibilité optimale

VEUILLEZ FOURNIR LA RÉPONSE DANS UNE CELLULE DE CODE.
"""

# User specifications template
USER_SPECIFICATIONS_TEMPLATE = """Spécifications supplémentaires:

{user_specifications}

Veuillez tenir compte de ces spécifications lors de la génération de votre réponse technique. Intégrez ces exigences particulières dans votre analyse et assurez-vous que la réponse finale respecte ces contraintes supplémentaires."""

# Keep the original template for backward compatibility
RESPONSE_TEMPLATE = SYSTEM_RESPONSE_TEMPLATE

# Welcome message template
WELCOME_MESSAGE_TEMPLATE = """Bienvenue ! Nous recherchons maintenant dans les deux collections: {collection_name} et {dce_collection}

Modèle LLM actif: *{llm_provider}*

**Ecrire le titre du paragraphe à générer.**"""

# Status messages
STATUS_MESSAGES = {
    "searching": "**Recherche en cours...**",
    "hybrid_search": "**Recherche hybride en cours dans {collection_name}...**",
    "search_complete": "**Recherche terminée. {total_docs} documents trouvés au-dessus du seuil de {threshold}. Affichage des résultats...**",
    "generating": "**Génération de la réponse en cours...**",
    "generation_based_on": "**Génération basée sur: {collections_used}**",
    "no_documents": "**Aucun document disponible pour générer une réponse.**",
    "no_results_threshold": "**La recherche hybride dans {collection_name} n'a retourné aucun résultat au-dessus du seuil de {threshold}.**",
    "fallback_retriever": "**Problème avec la recherche hybride dans {collection_name}. Utilisation du retriever par défaut...**",
    "hybrid_search_error": "**Problème avec la recherche hybride dans {collection_name}. Aucun résultat disponible.**",
    "no_documents_found": "**Aucun document n'a été trouvé avec un score de similarité supérieur à {threshold} dans les deux collections.**",
}

# Action labels
ACTION_LABELS = {
    "use_both": "Utiliser les deux collections",
    "use_technical_only": "Utiliser uniquement {collection_name}",
    "use_dce_only": "Utiliser uniquement {dce_collection}",
    "add_specifications": "Ajouter des spécifications",
    "do_not_generate": "Ne pas générer"
}

#  action labels for specifications
SPECS_ACTION_LABELS = {
    "use_both_with_specs": "Utiliser les deux collections (avec spécifications)",
    "use_technical_with_specs": "Utiliser uniquement {collection_name} (avec spécifications)",
    "use_dce_with_specs": "Utiliser uniquement {dce_collection} (avec spécifications)",
    "cancel_specs": "Annuler"
}

# Response choice message
RESPONSE_CHOICE_MESSAGE = "**{total_docs} documents trouvés au total avec un score supérieur à {threshold}. Choisissez la collection à utiliser pour la génération de réponse:**"

# Specifications input messages
SPECIFICATIONS_INPUT_MESSAGE = """**Veuillez entrer vos spécifications supplémentaires (techniques ou sur le style du pragraphe) :**

Ces spécifications seront intégrées à la génération de la réponse technique. 

"""

SPECIFICATIONS_COLLECTION_CHOICE_MESSAGE = """**Spécifications reçues :**

{specs_preview}

**Choisissez maintenant la collection à utiliser pour la génération avec vos spécifications :**"""

# Timeout and error messages
SPECIFICATIONS_TIMEOUT_MESSAGE = "**Timeout ou annulation.** Vous pouvez relancer l'action si nécessaire."
SPECIFICATIONS_EMPTY_MESSAGE = "**Aucune spécification fournie.** Vous pouvez relancer la génération sans spécifications."

# Cancellation message
CANCELLATION_MESSAGE = "La génération de réponse a été ignorée. Vous pouvez poser une nouvelle question."