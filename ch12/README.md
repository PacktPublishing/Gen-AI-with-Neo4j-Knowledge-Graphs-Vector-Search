# Deploying Your Haystack Chatbot to Google Cloud Run

This guide helps you containerize and deploy your Haystack chatbot to Google Cloud Run. The same Docker container can also be deployed on AWS, Azure, and other platforms using similar steps.

---

## Project Structure

```plaintext
haystack-cloud-app/
├── app.py
├── requirements.txt
├── Dockerfile
├── .env
├── 

## Dependencies

Add the following dependencies to your `requirements.txt` file:

```plaintext
haystack-ai==2.5.0
openai==1.67.0
gradio==4.44.1
python-dotenv>=1.0.0
neo4j==5.25.0
neo4j-haystack==2.0.3
```

## .env File

Create a `.env` file from `example.env`:

```bash
cp example.env .env
```

### Example `.env` Contents:

```plaintext
OPENAI_API_KEY=<insert-your-openai-api-key>
NEO4J_URI=<insert-your-neo4j-uri>
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<insert-your-neo4j-password>
```

## Dockerfile

Below is the `Dockerfile` for containerizing your application:

```dockerfile
FROM python:3.11

EXPOSE 8080

WORKDIR /app

COPY . ./

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
```

## Setting Up Google Cloud

### 1. Launch Cloud Shell
Go to the [Google Cloud Console](https://console.cloud.google.com/), and click on the terminal icon to open Cloud Shell.

### 2. Set Your Active Project
Run the following commands to set and verify your active project:
```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config list project
```

### 3. Enable Required Services
Enable the necessary Google Cloud services:
```bash
gcloud services enable cloudresourcemanager.googleapis.com \
                       servicenetworking.googleapis.com \
                       run.googleapis.com \
                       cloudbuild.googleapis.com \
                       cloudfunctions.googleapis.com
```

### 4. Upload or Clone Project Files
- **Option 1**: Upload files manually using the Cloud Shell Editor.
- **Option 2**: Clone the repository from GitHub:
  ```bash
  git clone https://github.com/PacktPublishing/Gen-AI-with-Neo4j-Knowledge-Graphs-Vector-Search.git
  cd Gen-AI-with-Neo4j-Knowledge-Graphs-Vector-Search/ch12
  ```

---

## Building and Deploying the Container

### 1. Set Environment Variables
Set the following environment variables:
```bash
export GCP_PROJECT='your-project-id'
export GCP_REGION='us-central1'
export AR_REPO='your-repo-name'
export SERVICE_NAME='movies-chatbot'
```

### 2. Create Artifact Registry Repository
Create a Docker repository in Artifact Registry:
```bash
gcloud artifacts repositories create "$AR_REPO" \
    --location="$GCP_REGION" \
    --repository-format=Docker
```

### 3. Authenticate Docker with Artifact Registry
Authenticate Docker to push images to Artifact Registry:
```bash
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"
```

### 4. Build and Push the Docker Image
Build and push your Docker image to Artifact Registry:
```bash
gcloud builds submit \
  --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"
```

### 5. Prepare Environment Variables for Deployment
Prepare the environment variables from your `.env` file:
```bash
ENV_VARS=$(grep -v '^#' .env | sed 's/ *= */=/g' | xargs -I{} echo -n "{},")
ENV_VARS=${ENV_VARS%,}
```

### 6. Deploy to Cloud Run
Deploy your containerized application to Google Cloud Run:
```bash
gcloud run deploy "$SERVICE_NAME" \
  --port=8080 \
  --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
  --allow-unauthenticated \
  --region=$GCP_REGION \
  --platform=managed \
  --project=$GCP_PROJECT \
  --set-env-vars="GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION,$ENV_VARS"
```

---

## Access Your Chatbot
Once deployed, access your chatbot at:
```
https://movies-chatbot-<unique-id>.<region>.run.app
```

---

## Deploying to Other Cloud Providers
You can reuse the same Docker setup to deploy on other platforms:
- **Azure**: Deploy Docker container to [Azure Container Apps](https://learn.microsoft.com/en-us/azure/container-apps/).
- **AWS**: Deploy Docker Compose to [AWS ECS](https://aws.amazon.com/ecs/).

---

## References for Spring Boot Application Deployment (Optional)
- **Google Cloud Run**: [Deploy Java service](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/java).
- **Azure Spring Apps**: [Deploy Maven-based apps](https://learn.microsoft.com/en-us/azure/spring-apps/).
- **AWS EC2 Deployment**: [Spring Boot on AWS EC2](https://aws.amazon.com/ec2/).
- **AWS Elastic Beanstalk**: [Spring Boot on Elastic Beanstalk](https://aws.amazon.com/elasticbeanstalk/).

✅ **Congratulations!** Your Haystack chatbot is now cloud-ready and deployable across platforms.
