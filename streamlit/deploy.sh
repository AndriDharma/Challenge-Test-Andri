#!/bin/bash

# This script deploys a containerized application to Google Cloud Run.
# It reads configuration variables from a .env file in the same directory.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration Loading ---
# Check if the .env file exists in the current directory.
if [ -f .env ]; then
  # If it exists, export the variables from the .env file to the shell's environment.
  # This command reads the file, ignores comments and empty lines, and exports the variables.
  export $(cat .env | sed 's/#.*//g' | xargs)
  echo "Configuration loaded from .env file."
else
  # If the .env file is not found, print an error and exit.
  echo "Error: .env file not found!"
  exit 1
fi

# --- Derived Variables ---
# Construct the full image URL for Artifact Registry from the loaded variables.
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}"

# Combine environment variables that will be set on the Cloud Run service.
ENV_VARS="PROJECT_ID=${PROJECT_ID},VERSION_ID=${VERSION_ID},SECRET_ID_DB=${SECRET_ID_DB}"


# --- Google Cloud CLI Operations ---
# Authenticate with Google Cloud. This might open a browser window for you to log in.
echo "Authenticating with Google Cloud..."
gcloud auth login

# 1. Set the active gcloud project.
echo "Setting GCP project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# 2. Build the container image using Cloud Build and push it to Artifact Registry.
echo "Building and submitting the container image..."
gcloud builds submit --region ${REGION} --tag ${IMAGE}

# 3. Deploy the container image to Cloud Run with the specified settings.
echo "Deploying the container to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE} \
  --platform managed \
  --region ${REGION} \
  --set-env-vars "${ENV_VARS}" \
  --port 8080 \
  --allow-unauthenticated \
  --service-account "${SERVICE_ACCOUNT}" \
  --min-instances ${MIN_INSTANCES} \
  --max-instances ${MAX_INSTANCES}
  # Note: If your service needs to connect to Cloud SQL, uncomment the following line:
  # --add-cloudsql-instances="${CLOUD_SQL_INSTANCE}"

# 4. Fetch and display the URL of the deployed service.
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')
echo "-----------------------------------------"
echo "Service deployed successfully!"
echo "Service URL: ${SERVICE_URL}"
echo "-----------------------------------------"
