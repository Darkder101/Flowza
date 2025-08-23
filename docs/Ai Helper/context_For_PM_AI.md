## 🏗️ Architecture

- **Backend**: FastAPI + SQLAlchemy + PostgreSQL
- **Task Queue**: Celery + Redis  
- **ML Execution**: Docker containers
- **Frontend**: React + React Flow (coming in Week 5-6)

## 🛠️ MVP Roadmap (6–8 Weeks)
### Phase 1: Foundation + Proof of Concept (Weeks 1–2)
- Setup FastAPI backend
- Define workflow + task model
- Implement task runner (Celery + Redis)
- Add 2–3 basic ML tasks:
    - Load CSV
    - Drop null values
    - Train/Test Split

### Phase 2: Add Core ML Tools (Weeks 3–4)
- Add ML nodes:
    - Preprocessing (normalize, encode categorical, feature selection)
    - Training (Logistic Regression, XGBoost)
    - Evaluation (accuracy, F1 score)
- Workflow chaining (output → input passing via Python objects)
- Store workflows + results in PostgreSQL

### Phase 3: Visual Interface (Weeks 5–6)
- React + React Flow frontend
- Drag-and-drop workflow editor
- Connect frontend to backend (API to run workflows)
- Display outputs (tables, metrics, charts)

### Phase 4: AI + Workflow Templates (Weeks 7–8)
- Add simple AI-assisted workflow generation:
    - Prompt → generate workflow JSON → render in React Flow
- Add workflow templates:
    - CSV → Clean → Split → Train Logistic Regression → Evaluate
    - CSV → Normalize → Train XGBoost → Export model
- Export option: Generate Python script / notebook from workflow

## 📅 Development Roadmap

- [x] **Week 1**: Project setup, basic infrastructure
- [ ] **Week 2**: Workflow execution engine  
- [ ] **Week 3-4**: Core ML nodes (preprocess, train, evaluate)
- [ ] **Week 5-6**: React frontend with visual workflow editor
- [ ] **Week 7-8**: AI-assisted workflow generation

### 📋 Detailed Week-by-Week Plan
#### Week 1: Project Setup
- [x] Setup FastAPI backend, Docker, PostgreSQL, Redis, Celery
- [x] Define schema for workflows + tasks
- [x] Create first tool: CSV Loader

#### Week 2: Basic Execution Engine
- [ ] Implement workflow execution service
- [ ] Chain simple nodes: CSV → Drop Nulls → Output Dataset
- [ ] Build APIs for workflow submission + result fetching

#### Week 3: Add Preprocessing Tools
- [ ] Normalize values, one-hot encoding, train/test split
- [ ] Save intermediate datasets in PostgreSQL/S3 (depending on size)
- [ ] Testing basic pipelines

#### Week 4: Add Training & Eval
- [ ] Logistic Regression (scikit-learn)
- [ ] XGBoost Classifier
- [ ] Evaluation node (accuracy, F1 score, confusion matrix)

#### Week 5: Frontend Workflow Builder
- [ ] React + React Flow setup
- [ ] Add workflow canvas
- [ ] Nodes: Data → Preprocess → Train → Evaluate
- [ ] Connect backend API to frontend run button

#### Week 6: Workflow Output UI
- [ ] Show metrics in UI (tables, charts)
- [ ] Display trained model artifacts for download
- [ ] Save + load workflows from DB

#### Week 7: AI Assistance + Templates
- [ ] Integrate OpenAI API → “Generate workflow from prompt”
- [ ] Prebuilt templates (classification, regression, data cleaning)

#### Week 8: Polish MVP
- [ ] Export workflows as Python notebooks
- [ ] Test with example datasets (Titanic, Iris)
- [ ] Docs + Demo-ready

## Project Structure
```
MLFlowBuilder/
├── .env
├── .gitignore
├── docker-compose.yml
├── README.md
├── requirements.txt
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── database.py
|   |   |   └── connection.py 
│   │   ├── models/
|   |   |   ├── dataset.py 
│   │   │   ├── workflow.py
│   │   │   └── task.py
│   │   ├── routers/
│   │   │   ├── workflows.py
│   │   │   └── tasks.py
│   │   ├── services/
|   |   |   ├── task_executor.py 
│   │   │   ├── workflow_service.py
│   │   │   └── ml_nodes/
│   │   │       ├── base.py
│   │   │       ├── csv_loader.py
│   │   │       ├── preprocess.py
│   │   │       ├── train_logreg.py
│   │   │       ├── train_xgboost.py
│   │   │       └── evaluate.py
│   │   ├── schemas/
│   │   │   ├── workflow_schemas.py
│   │   │   └── task_schemas.py
│   │   └── ai/
│   │       └── workflow_generator.py
|   ├── datasets/
|   |   ├── iris.csv
|   |   ├── sample_data.csv
|   |   └── housing.csv
|   ├── models/
|   ├── scripts/
|   |   ├── start_celery.sh
|   |   └── start_dev.sh
│   ├── requirements.txt
│   ├── init.sql
|   └── .env
├── frontend/  
├── docs/
│   ├── Ai_Helper/
|   ├── dev_journal/
|   ├── guide/
└── docker/
    └── ml-base.dockerfile
```

## Flowza Development Log
### Day 1: Project Setup & Infrastructure
✅ Progress
- Set up full FastAPI backend structure with organized models, routers, schemas, and services
- Configured VS Code workspace + Python venv with dependencies
- Added Docker setup (docker-compose.yml, ML-base.dockerfile)
- Implemented database connection and models (workflow, dataset)
- Built API routers for workflows and tasks, tested endpoints
- Created ML node system: base class, CSV loader, preprocessing node
- Added task executor service
- Organized sample datasets (iris.csv, housing.csv)
- Added scripts (start_dev.sh, start_celery.sh)
- Updated .gitignore


Based on the roadmap and logs, continue with planning and execution from today , you are going to be my project manager not developer. your task will be to guide to build MVP project according to plan and detailes i have given. in detailed instructions day-to-day. i will give you idea how you can structure of day-to-day instructions file.

# Flowza Day 1: Project Setup & Infrastructure

## Day 1 objectives : 
## Step 1: Project Structure Setup
### 1.1 Create the base project structure
### 1.2 Create configuration files
## Step 2: VS Code Workspace Configuration
### 2.1 Create VS Code workspace file
### 2.2 Open workspace in VS Code
## Step 3: Backend Environment Setup
### 3.1 Create Python virtual environment
### 3.2 Install core dependencies
Create `backend/requirements.txt`:
### 3.3 Create environment configuration
Create `backend/.env`:
## Step 4: Database & Docker Setup
### 4.1 Create docker-compose.yml
In the root directory, create `docker-compose.yml`:
### 4.2 Create ML base Docker image
Create `docker/ml-base.dockerfile`:
## Step 5: Database Models & Connection
### 5.1 Create database connection
Create `backend/app/database/connection.py`:
### 5.2 Create core models
Create `backend/app/models/workflow.py`:
Create `backend/app/models/task.py`:
Create `backend/app/models/dataset.py`:
## Step 6: Basic FastAPI Application
### 6.1 Create main FastAPI application
Create `backend/app/main.py`:
### 6.2 Create basic routers
Create `backend/app/routers/workflows.py`:
Create `backend/app/routers/tasks.py`:
### 6.3 Create Pydantic schemas
Create `backend/app/schemas/workflow_schemas.py`:
Create `backend/app/schemas/task_schemas.py`:
## Step 7: Create First ML Node
### 7.1 Create base ML node class
Create `backend/app/services/ml_nodes/base.py`:
### 7.2 Create CSV Loader node
Create `backend/app/services/ml_nodes/csv_loader.py`:
### 7.3 Create basic preprocessing node
Create `backend/app/services/ml_nodes/preprocess.py`:
### 7.4 Create task executor service
Create `backend/app/services/task_executor.py`:
## Step 8: Test the Setup
### 8.1 Run the FastAPI server
### 8.2 Test the API endpoints
### 8.3 Test with curl commands
### 8.4 Test file upload
## Step 9: Create Sample Datasets
### 9.1 Create sample datasets directory
### 9.2 Test CSV loader with sample data
## 🔜 Next Steps (Week 2)
Tomorrow you'll implement:

 you dont have to follow exact same structure as this but almost like this and also map which files each step will touch/change.