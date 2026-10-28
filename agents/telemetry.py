"""Tracing to Azure Monitor / Application Insights (optional).

When APPLICATIONINSIGHTS_CONNECTION_STRING is set, the Agent Framework's OpenTelemetry spans
(workflow, executors, agent runs, model calls with token usage) are exported to Application
Insights. Connect the same Application Insights resource to the Foundry project and the traces
appear under Tracing in the Foundry portal. Without the variable this module does nothing.

ENABLE_SENSITIVE_DATA=true also records prompts and completions (off by default).
"""
from __future__ import annotations

import os

_DONE = False


def setup_tracing(service_name: str = "chance-genesis") -> bool:
    global _DONE
    cs = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if _DONE or not cs:
        return _DONE
    from agent_framework.observability import configure_otel_providers
    from azure.monitor.opentelemetry.exporter import (
        AzureMonitorLogExporter, AzureMonitorMetricExporter, AzureMonitorTraceExporter,
    )
    configure_otel_providers(
        service_name=service_name,
        exporters=[AzureMonitorTraceExporter(connection_string=cs),
                   AzureMonitorLogExporter(connection_string=cs),
                   AzureMonitorMetricExporter(connection_string=cs)],
        enable_sensitive_data=os.getenv("ENABLE_SENSITIVE_DATA", "").lower() == "true",
    )
    _DONE = True
    return True
