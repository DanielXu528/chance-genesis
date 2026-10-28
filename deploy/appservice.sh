#!/usr/bin/env bash
# Deploy Chance Genesis to Azure App Service (Linux, Python 3.12) with keyless access to Microsoft Foundry.
# Works on subscriptions without container builds (e.g. Azure for Students). Run in Azure Cloud Shell or
# anywhere with the Azure CLI logged in. No build step on the platform: on each start the app pulls the
# code from the public GitHub repository, installs the dependencies into /home/site/venv once, builds the
# demo match once into /home/site/data (persistent), then serves.
#
#   export FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
#   export FOUNDRY_MODEL=gpt-5-mini
#   export FOUNDRY_PROJECT_ID=<project resource id>          # to grant the app the Foundry User role
#   export APPLICATIONINSIGHTS_CONNECTION_STRING=...          # optional, tracing
#   ./deploy/appservice.sh
#
# First start takes about 15 minutes on B2 (dependencies + building the 90-minute match); restarts take
# well under a minute. Push to the repository and restart the app to deploy a new version.
set -euo pipefail
RG=${RG:-rg-chance-genesis}
LOC=${LOC:-westus}
PLAN=${PLAN:-cg-plan}
SKU=${SKU:-B2}
WEB=${WEB:-chance-genesis-$RANDOM}
REPO_TARBALL=${REPO_TARBALL:-https://github.com/DanielXu528/chance-genesis/archive/refs/heads/main.tar.gz}
FOUNDRY_USER_ROLE=53ca6127-db72-4b80-b1b0-d745d6d5456d

az group create -n "$RG" -l "$LOC" -o none
az appservice plan show -n "$PLAN" -g "$RG" -o none 2>/dev/null || \
  az appservice plan create -n "$PLAN" -g "$RG" -l "$LOC" --is-linux --sku "$SKU" -o none
az webapp show -n "$WEB" -g "$RG" -o none 2>/dev/null || \
  az webapp create -n "$WEB" -g "$RG" -p "$PLAN" --runtime PYTHON:3.12 -o none

SETTINGS=(WEBSITES_CONTAINER_START_TIME_LIMIT=1800 CG_DATA_DIR=/home/site/data/demo SCM_DO_BUILD_DURING_DEPLOYMENT=false)
[[ -n "${FOUNDRY_PROJECT_ENDPOINT:-}" ]] && SETTINGS+=(FOUNDRY_PROJECT_ENDPOINT="$FOUNDRY_PROJECT_ENDPOINT" FOUNDRY_MODEL="${FOUNDRY_MODEL:-gpt-5-mini}")
[[ -n "${APPLICATIONINSIGHTS_CONNECTION_STRING:-}" ]] && SETTINGS+=(APPLICATIONINSIGHTS_CONNECTION_STRING="$APPLICATIONINSIGHTS_CONNECTION_STRING")
az webapp config appsettings set -n "$WEB" -g "$RG" --settings "${SETTINGS[@]}" -o none

if [[ -n "${FOUNDRY_PROJECT_ENDPOINT:-}" ]]; then
  PRINCIPAL=$(az webapp identity assign -n "$WEB" -g "$RG" --query principalId -o tsv)
  if [[ -n "${FOUNDRY_PROJECT_ID:-}" ]]; then
    az role assignment create --role "$FOUNDRY_USER_ROLE" --assignee-object-id "$PRINCIPAL" \
      --assignee-principal-type ServicePrincipal --scope "$FOUNDRY_PROJECT_ID" -o none
  else
    echo "Grant the Foundry User role ($FOUNDRY_USER_ROLE) on the project to principal $PRINCIPAL"
  fi
fi

START="mkdir -p /home/site/app && cd /home/site/app && curl -sL $REPO_TARBALL | tar xz --strip-components=1"
START+=" && if [ ! -x /home/site/venv/bin/uvicorn ]; then python -m venv /home/site/venv && /home/site/venv/bin/pip install -q -r requirements.txt; fi"
START+=" && export PATH=/home/site/venv/bin:\$PATH && exec bash startup.sh"
az webapp config set -n "$WEB" -g "$RG" --always-on true --startup-file "bash -c \"$START\"" -o none
az webapp restart -n "$WEB" -g "$RG" -o none
echo "URL: https://$(az webapp show -n "$WEB" -g "$RG" --query defaultHostName -o tsv)"
