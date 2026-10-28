#!/usr/bin/env bash
# Deploy Chance Genesis to Azure App Service (Linux, Python 3.12) with keyless access to Microsoft Foundry.
# Use this on subscriptions where container builds (ACR Tasks) are not available, e.g. Azure for Students;
# otherwise deploy/aca.sh (Azure Container Apps) works too. Run from the repository root, e.g. in Azure Cloud Shell.
#
#   export FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
#   export FOUNDRY_MODEL=gpt-5-mini
#   export FOUNDRY_PROJECT_ID=<project resource id>          # to grant the app the Foundry User role
#   export APPLICATIONINSIGHTS_CONNECTION_STRING=...          # optional, tracing
#   ./deploy/appservice.sh
set -euo pipefail
RG=${RG:-rg-chance-genesis}
LOC=${LOC:-westus}
PLAN=${PLAN:-cg-plan}
WEB=${WEB:-chance-genesis-$RANDOM}
FOUNDRY_USER_ROLE=53ca6127-db72-4b80-b1b0-d745d6d5456d

az group create -n "$RG" -l "$LOC" -o none
az appservice plan show -n "$PLAN" -g "$RG" -o none 2>/dev/null || \
  az appservice plan create -n "$PLAN" -g "$RG" -l "$LOC" --is-linux --sku B1 -o none
az webapp show -n "$WEB" -g "$RG" -o none 2>/dev/null || \
  az webapp create -n "$WEB" -g "$RG" -p "$PLAN" --runtime PYTHON:3.12 -o none
az webapp config set -n "$WEB" -g "$RG" --startup-file "bash startup.sh" --always-on true -o none
SETTINGS=(SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITES_CONTAINER_START_TIME_LIMIT=900)
[[ -n "${FOUNDRY_PROJECT_ENDPOINT:-}" ]] && SETTINGS+=(FOUNDRY_PROJECT_ENDPOINT="$FOUNDRY_PROJECT_ENDPOINT" FOUNDRY_MODEL="${FOUNDRY_MODEL:-gpt-5-mini}")
[[ -n "${APPLICATIONINSIGHTS_CONNECTION_STRING:-}" ]] && SETTINGS+=(APPLICATIONINSIGHTS_CONNECTION_STRING="$APPLICATIONINSIGHTS_CONNECTION_STRING")
az webapp config appsettings set -n "$WEB" -g "$RG" --settings "${SETTINGS[@]}" -o none
if [[ -n "${FOUNDRY_PROJECT_ENDPOINT:-}" ]]; then
  PRINCIPAL=$(az webapp identity assign -n "$WEB" -g "$RG" --query principalId -o tsv)
  if [[ -n "${FOUNDRY_PROJECT_ID:-}" ]]; then
    az role assignment create --role "$FOUNDRY_USER_ROLE" --assignee-object-id "$PRINCIPAL" \
      --assignee-principal-type ServicePrincipal --scope "$FOUNDRY_PROJECT_ID" -o none
    echo "Granted Foundry User on the project to the app's managed identity."
  fi
fi
rm -f /tmp/chance-genesis.zip
git archive --format=zip -o /tmp/chance-genesis.zip HEAD
az webapp deploy -n "$WEB" -g "$RG" --src-path /tmp/chance-genesis.zip --type zip -o none
echo "URL: https://$(az webapp show -n "$WEB" -g "$RG" --query defaultHostName -o tsv)  (first start builds the demo match, about 3 minutes)"
