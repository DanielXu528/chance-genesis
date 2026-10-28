#!/usr/bin/env bash
# Deploy Chance Genesis to Azure Container Apps.
# Prereqs: Azure CLI 2.80+ logged in (az login), containerapp extension (az extension add -n containerapp).
#
# Narration backend (pick one, or none for the offline template writer):
#   A) Microsoft Foundry project (recommended, keyless):
#        export FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
#        export FOUNDRY_MODEL=<deployment name, e.g. gpt-5-mini>
#        export FOUNDRY_PROJECT_ID=$(az cognitiveservices account project show -n <resource> -g <rg> \
#                                     --project-name <project> --query id -o tsv)   # optional
#      The app authenticates with its managed identity. With FOUNDRY_PROJECT_ID set, this script grants
#      that identity the Foundry User role on the project; otherwise it prints the command.
#   B) Azure OpenAI:   export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
#                      export AZURE_OPENAI_DEPLOYMENT=<deployment name>
#                      export AZURE_OPENAI_API_KEY=<key>   (optional; without it the app uses its managed
#                      identity, which needs "Cognitive Services OpenAI User")
# Tracing (optional): export APPLICATIONINSIGHTS_CONNECTION_STRING=<connection string>
#   Use the Application Insights resource connected to the Foundry project to see the traces under
#   Tracing in the Foundry portal.
set -euo pipefail
RG=${RG:-rg-chance-genesis}
LOC=${LOC:-eastus}
APP=${APP:-chance-genesis}
FOUNDRY_USER_ROLE=53ca6127-db72-4b80-b1b0-d745d6d5456d   # "Foundry User" (formerly "Azure AI User")

az group create -n "$RG" -l "$LOC" -o none
az containerapp up -n "$APP" -g "$RG" -l "$LOC" --source . --ingress external --target-port 8000

if [[ -n "${APPLICATIONINSIGHTS_CONNECTION_STRING:-}" ]]; then
  az containerapp secret set -n "$APP" -g "$RG" --secrets appi-cs="$APPLICATIONINSIGHTS_CONNECTION_STRING" -o none
  az containerapp update -n "$APP" -g "$RG" -o none \
    --set-env-vars APPLICATIONINSIGHTS_CONNECTION_STRING=secretref:appi-cs
fi

if [[ -n "${FOUNDRY_PROJECT_ENDPOINT:-}" ]]; then
  az containerapp identity assign -n "$APP" -g "$RG" --system-assigned -o none
  az containerapp update -n "$APP" -g "$RG" -o none \
    --set-env-vars FOUNDRY_PROJECT_ENDPOINT="$FOUNDRY_PROJECT_ENDPOINT" FOUNDRY_MODEL="${FOUNDRY_MODEL:-gpt-5-mini}"
  PRINCIPAL=$(az containerapp show -n "$APP" -g "$RG" --query identity.principalId -o tsv)
  if [[ -n "${FOUNDRY_PROJECT_ID:-}" ]]; then
    az role assignment create --role "$FOUNDRY_USER_ROLE" --assignee-object-id "$PRINCIPAL" \
      --assignee-principal-type ServicePrincipal --scope "$FOUNDRY_PROJECT_ID" -o none
    echo "Granted Foundry User on the project to the app's managed identity ($PRINCIPAL)."
  else
    echo "Grant the app's managed identity the Foundry User role on your Foundry project:"
    echo "  az role assignment create --role $FOUNDRY_USER_ROLE --assignee-object-id $PRINCIPAL \\"
    echo "    --assignee-principal-type ServicePrincipal --scope <project resource id>"
  fi
elif [[ -n "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
  if [[ -n "${AZURE_OPENAI_API_KEY:-}" ]]; then
    az containerapp secret set -n "$APP" -g "$RG" --secrets aoai-key="$AZURE_OPENAI_API_KEY" -o none
    az containerapp update -n "$APP" -g "$RG" -o none \
      --set-env-vars AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
                     AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-5-mini}" \
                     AZURE_OPENAI_API_KEY=secretref:aoai-key
  else
    # keyless: managed identity; grant it "Cognitive Services OpenAI User" on the Azure OpenAI resource
    az containerapp identity assign -n "$APP" -g "$RG" --system-assigned -o none
    az containerapp update -n "$APP" -g "$RG" -o none \
      --set-env-vars AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
                     AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-5-mini}"
    echo "Grant 'Cognitive Services OpenAI User' to principal $(az containerapp show -n "$APP" -g "$RG" --query identity.principalId -o tsv)"
  fi
fi

echo "URL: https://$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
