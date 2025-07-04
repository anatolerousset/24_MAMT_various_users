# Assistant Mémoire Technique - Chainlit Interface

### Description
L'Assistant Mémoire Technique est une interface de chat alimentée par l'IA qui permet de rechercher et générer du contenu à partir de documents techniques et de DCE (Dossier de Consultation des Entreprises). L'application utilise Chainlit pour l'interface utilisateur et un backend FastAPI pour le traitement des requêtes.

### Fonctionnalités principales

#### 🔍 **Recherche hybride intelligente**
- Recherche simultanée dans deux collections de documents :
  - **Collection Technique** : Mémoires techniques et documents spécialisés
  - **Collection DCE** : Dossiers de consultation des entreprises
- Utilisation de la recherche vectorielle avec re-classement automatique
- Affichage des scores de pertinence et sources

#### 💬 **Génération de contenu personnalisée**
- Génération de paragraphes techniques basée sur les documents trouvés
- Possibilité d'ajouter des spécifications personnalisées
- Choix flexible des collections à utiliser (une seule ou les deux)
- Streaming en temps réel du contenu généré

#### 📤 **Export de documents**
- Export au format DOCX du paragraphe généré uniquement
- Export complet incluant tous les documents récupérés
- Téléchargement direct des fichiers générés
- Inclusion automatique d'un catalogue d'images

### Guide d'utilisation

#### 1. **Démarrage de l'application**
```bash
# Lancer l'interface Chainlit
chainlit run chainlit_frontend.py

# L'application sera accessible sur http://localhost:8000
```

#### 2. **Vérification du système**
Au démarrage, l'interface affiche :
- ✅ Statut des services (base vectorielle, modèle de langage, stockage)
- 📊 Collections disponibles et configuration LLM
- 📁 Lien vers l'interface d'ingestion Streamlit

#### 3. **Processus de recherche**
1. **Saisir votre requête** : Tapez le titre du paragraphe que vous souhaitez générer
2. **Recherche automatique** : Le système lance une recherche hybride avec barre de progression
3. **Résultats affichés** : Visualisation des documents trouvés avec scores et métadonnées

#### 4. **Personnalisation**
Après la recherche, le système demande :
- **Spécifications supplémentaires** : Ajoutez des critères techniques ou de style
- Tapez "aucunes" si aucune spécification particulière

#### 5. **Sélection des collections**
Choisissez parmi :
- 2️⃣ **Utiliser les deux collections** (recommandé)
- 📚 **Collection MT uniquement** (mémoires techniques)
- 📋 **Collection DCE uniquement** (dossiers de consultation)

#### 6. **Génération et export**
- Le contenu est généré en streaming temps réel
- Options d'export disponibles après génération :
  - 📄 **Paragraphe seul** (DOCX)
  - 📑 **Document complet** avec sources (DOCX)

### 🔄 **Ingestion de nouveaux documents**

Pour ajouter de nouveaux documents au système :

#### Accès à l'interface Streamlit
1. **URL d'accès** : `http://localhost:8501` (lien affiché dans l'interface Chainlit)
2. **Interface d'ingestion** : Streamlit permet l'upload et le traitement de nouveaux fichiers
3. **Formats supportés** : PDF, DOCX, et autres formats de documents techniques

#### Processus d'ingestion
1. Accédez à l'interface Streamlit via le lien fourni
2. Sélectionnez la collection cible (Technique ou DCE)
3. Uploadez vos documents
4. Configurez les paramètres de traitement si nécessaire
5. Lancez l'ingestion
6. Les nouveaux documents seront immédiatement disponibles dans Chainlit

### Configuration

#### Variables d'environnement
```bash
BACKEND_URL=http://backend:8001          # URL du backend FastAPI
IMAGES_URL=http://localhost:8001         # URL pour servir les images
CHAINLIT_HOST=localhost                  # Host de l'interface Chainlit
```

#### Structure des collections
- **Collection Technique** : Documents spécialisés, guides techniques
- **Collection DCE** : Cahiers des charges, spécifications projet

### Dépannage

#### Problèmes courants
- **🔴 Système non disponible** : Vérifiez que le backend est démarré
- **⏱️ Timeout** : La recherche peut prendre du temps, réessayez
- **❌ Aucun document trouvé** : Reformulez votre requête ou vérifiez les collections

#### Logs et monitoring
- Les logs sont affichés dans la console du serveur
- Statut des services visible au démarrage de l'application

---

## English Version

### Description
The Technical Memory Assistant is an AI-powered chat interface that enables searching and generating content from technical documents and tender specification files (DCE). The application uses Chainlit for the user interface and a FastAPI backend for request processing.

### Key Features

#### 🔍 **Intelligent Hybrid Search**
- Simultaneous search across two document collections:
  - **Technical Collection**: Technical specifications and specialized documents
  - **DCE Collection**: Tender and consultation documents
- Vector search with automatic reranking
- Relevance scores and source display

#### 💬 **Personalized Content Generation**
- Technical paragraph generation based on retrieved documents
- Option to add custom specifications
- Flexible collection selection (single or both)
- Real-time content streaming

#### 📤 **Document Export**
- DOCX export of generated paragraph only
- Complete export including all retrieved documents
- Direct file download
- Automatic image catalog inclusion

### User Guide

#### 1. **Starting the Application**
```bash
# Launch Chainlit interface
chainlit run chainlit_frontend.py

# Application will be accessible at http://localhost:8000
```

#### 2. **System Status Check**
At startup, the interface displays:
- ✅ Service status (vector database, language model, storage)
- 📊 Available collections and LLM configuration
- 📁 Link to Streamlit ingestion interface

#### 3. **Search Process**
1. **Enter your query**: Type the title of the paragraph you want to generate
2. **Automatic search**: System launches hybrid search with progress bar
3. **Results displayed**: View found documents with scores and metadata

#### 4. **Customization**
After search, the system requests:
- **Additional specifications**: Add technical or style criteria
- Type "aucunes" (none) if no particular specifications

#### 5. **Collection Selection**
Choose from:
- 2️⃣ **Use both collections** (recommended)
- 📚 **Technical collection only**
- 📋 **DCE collection only**

#### 6. **Generation and Export**
- Content is generated with real-time streaming
- Export options available after generation:
  - 📄 **Paragraph only** (DOCX)
  - 📑 **Complete document** with sources (DOCX)

### 🔄 **New Document Ingestion**

To add new documents to the system:

#### Streamlit Interface Access
1. **Access URL**: `http://localhost:8501` (link displayed in Chainlit interface)
2. **Ingestion interface**: Streamlit enables file upload and processing
3. **Supported formats**: PDF, DOCX, and other technical document formats

#### Ingestion Process
1. Access the Streamlit interface via the provided link
2. Select target collection (Technical or DCE)
3. Upload your documents
4. Configure processing parameters if necessary
5. Launch ingestion
6. New documents will be immediately available in Chainlit

### Configuration

#### Environment Variables
```bash
BACKEND_URL=http://backend:8001          # FastAPI backend URL
IMAGES_URL=http://localhost:8001         # URL for serving images
CHAINLIT_HOST=localhost                  # Chainlit interface host
```

#### Collection Structure
- **Technical Collection**: Specialized documents, technical guides
- **DCE Collection**: Specifications, project requirements

### Troubleshooting

#### Common Issues
- **🔴 System unavailable**: Check that backend is running
- **⏱️ Timeout**: Search may take time, please retry
- **❌ No documents found**: Rephrase your query or check collections

#### Logs and Monitoring
- Logs are displayed in server console
- Service status visible at application startup

### Technical Requirements
- Python 3.8+
- Chainlit
- FastAPI backend
- Vector database (Qdrant)
- Language model service

### Support
For technical issues or questions about document ingestion, use the Streamlit interface accessible via the link provided in the Chainlit application.