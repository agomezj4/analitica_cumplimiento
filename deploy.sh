#!/bin/bash

ENVIRONMENT=$1

if [ "$ENVIRONMENT" = "staging" ]; then
    echo "Deploying en staging..."
    # Comandos específicos para el despliegue en staging, por ejemplo:
    # rsync -avz --exclude 'env' . user@staging-server:/path/to/deploy/
elif [ "$ENVIRONMENT" = "production" ]; then
    echo "Deploying en producción..."
    # Comandos específicos para el despliegue en producción, por ejemplo:
    # rsync -avz --exclude 'env' . user@production-server:/path/to/deploy/
else
    echo "Unknown environment: $ENVIRONMENT"
    exit 1
fi
