![FastAPI](https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png)


# Face Recognitation API

The Face Recognitation API consists of two main components: a database hosted on Supabase and the API itself hosted on an Amazon EC2 instance. Additionally, the API can be run locally using a Docker container.

## A) Amazon EC2 Deployment

### Accessing the Subway API
- The Subway API is hosted on an Amazon EC2 instance.
- Access it via the public IP endpoint: [http://50.19.149.26](http://50.19.149.26).

## B) Local Setup

### 1. Clone the GitHub Repository
```bash
git clone https://github.com/leon7731/Face-Recognitation-System-Model-API.git . 
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
- Obtain the `.env` file via email or download from the repository.
- Place the `.env` file in the Config directory.

### 4. Running the API Locally
- To run the Fast API server locally:
```bash
uvicorn App.Main:app
```
- The server will start on [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## C) Docker Deployment

### 1. Initial Setup
- Repeat steps 1 and 2 from the Local Setup.

### 2. Environment and SQL File Configuration
- Locate the `.env` and `init.sql` files.
- Move them into the appropriate directories.


### 3. Docker Commands
- Build and run the Docker container:
```bash
docker-compose up -d
```
