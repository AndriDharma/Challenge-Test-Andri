#!/bin/bash

# This script deploys a containerized application to Google Cloud Run.
# It reads configuration variables from a .env file in the same directory.

# Exit immediately if a command exits with a non-zero status.
set -e
# --- Configuration Loading ---
# Check if .env file exists and load variables from it.
if [ -f "$(dirname "$0")/.env" ]; then
  echo "Loading environment from .env"
  set -a
  source "$(dirname "$0")/.env"
  set +a
else
  echo "Error: .env file not found in $(pwd)"
  exit 1
fi

# --- Derived Variables ---
# Construct the full image path using variables from the .env file.
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}"

# Combine environment variables into a single string for the Cloud Run deployment command.
ENV_VARS="PROJECT_ID=$PROJECT_ID,VERSION_ID=$VERSION_ID,BUCKET_NAME=$BUCKET_NAME,SECRET_ID_DB=$SECRET_ID_DB"

# --- Execution ---
echo "--- Starting Deployment ---"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "Image: $IMAGE"
echo "--------------------------"

# Authenticate with Google Cloud. This might open a browser window for you to log in.
echo "Authenticating with Google Cloud..."
gcloud auth login

# Set the active gcloud project.
echo "Setting active project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"

# Build the container image using Cloud Build and tag it.
echo "Building the container image..."
gcloud builds submit --region "$REGION" --tag "$IMAGE"

# Deploy the container image to Cloud Run with the specified configuration.
echo "Deploying the container to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --set-env-vars "$ENV_VARS" \
  --port 8080 \
  --allow-unauthenticated \
  --service-account "$SERVICE_ACCOUNT" \
  --min-instances "$MIN_INSTANCES" \
  --max-instances "$MAX_INSTANCES" \
  --add-cloudsql-instances "$CLOUD_SQL_INSTANCE"

# --- Completion ---
# Retrieve the URL of the deployed service.
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format 'value(status.url)')
echo "✅ Deployment successful!"
echo "Service deployed to: $SERVICE_URL"
