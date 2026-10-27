#!/usr/bin/env bash
# Deploy Chance Genesis to Azure Container Apps.
# Prereqs: Azure CLI logged in (az login), containerapp extension (az extension add -n containerapp).
#
# Narration backend (pick one, or none for the offline template writer):
#   A) Microsoft Foundry project:  export FOUNDRY_PROJECT_ENDPOINT=https://<project>.services.ai.azure.com/api/projects/<name>
#                                  export FOUNDRY_MODEL=<deployment name, e.g. gpt-4.1-mini>
#      The app authenticates with its managed identity; grant it the "Azure AI User" role on the project.
#   B) Azure OpenAI:               export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
#                                  export AZURE_OPENAI_DEPLOYMENT=<deployment name>
#                                  export AZURE_OPENAI_API_KEY=<key>   (optional; without it the app uses
#                                  its managed identity, which needs "Cognitive Services OpenAI User")
set -euo pipefail
RG=${RG:-rg-chance-genesis}
LOC=${LOC:-eastus}
APP=${APP:-chance-genesis}

az group create -n "$RG" -l "$LOC" -o none
az containerapp up -n "$APP" -g "$RG" -l "$LOC" --source . --ingress external --target-port 8000

if [[ -n "${FOUNDRY_PROJECT_ENDPOINT:-}" ]]; then
  az containerapp identity assign -n "$APP" -g "$RG" --system-assigned -o none
  az containerapp update -n "$APP" -g "$RG" -o none \
    --set-env-vars FOUNDRY_PROJECT_ENDPOINT="$FOUNDRY_PROJECT_ENDPOINT" FOUNDRY_MODEL="${FOUNDRY_MODEL:-gpt-4.1-mini}"
  echo "Grant the app's managed identity the 'Azure AI User' role on your Foundry project:"
  echo "  principal: $(az containerapp show -n "$APP" -g "$RG" --query identity.principalId -o tsv)"
elif [[ -n "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
  if [[ -n "${AZURE_OPENAI_API_KEY:-}" ]]; then
    az containerapp secret set -n "$APP" -g "$RG" --secrets aoai-key="$AZURE_OPENAI_API_KEY" -o none
    az containerapp update -n "$APP" -g "$RG" -o none \
      --set-env-vars AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
                     AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-4.1-mini}" \
                     AZURE_OPENAI_API_KEY=secretref:aoai-key
  else
    # keyless: managed identity; grant it "Cognitive Services OpenAI User" on the Azure OpenAI resource
    az containerapp identity assign -n "$APP" -g "$RG" --system-assigned -o none
    az containerapp update -n "$APP" -g "$RG" -o none \
      --set-env-vars AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
                     AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-4.1-mini}"
    echo "Grant 'Cognitive Services OpenAI User' to principal $(az containerapp show -n "$APP" -g "$RG" --query identity.principalId -o tsv)"
  fi
fi

echo "URL: https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
