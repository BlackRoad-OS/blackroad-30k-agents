"""
Tests for the BlackRoad Base Agent Framework.
Covers: enums, BaseAgent core logic, all specialized agents, and the factory.
"""

import pytest
import time
from datetime import datetime

from base_agent import (
    AgentStatus,
    AgentType,
    BaseAgent,
    QuantumMechanicsAgent,
    CodeReviewAgent,
    TestingAgent,
    ResearchAgent,
    DocumentationAgent,
    MonitoringAgent,
    IntegrationAgent,
    AnalyticsAgent,
    DeploymentAgent,
    AGENT_REGISTRY,
    create_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConcreteAgent(BaseAgent):
    """Minimal concrete agent used for BaseAgent tests."""

    def __init__(self, agent_id=None):
        super().__init__(
            agent_type=AgentType.TESTING,
            capabilities=["unit_test_generation"],
            agent_id=agent_id,
        )

    def execute_task(self, task):
        action = task.get("action")
        if action == "raise":
            raise RuntimeError("deliberate failure")
        return {"echo": task.get("payload")}


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestAgentStatusEnum:
    def test_all_statuses_present(self):
        expected = {"initializing", "idle", "working", "error", "healing", "shutdown"}
        assert {s.value for s in AgentStatus} == expected

    def test_access_by_name(self):
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.SHUTDOWN.value == "shutdown"


class TestAgentTypeEnum:
    def test_all_types_present(self):
        expected = {
            "quantum_mechanics", "relativity", "cosmology",
            "code_review", "testing", "deployment", "research",
            "documentation", "monitoring", "integration", "analytics",
        }
        assert {t.value for t in AgentType} == expected

    def test_access_by_name(self):
        assert AgentType.CODE_REVIEW.value == "code_review"
        assert AgentType.ANALYTICS.value == "analytics"


# ---------------------------------------------------------------------------
# BaseAgent core tests
# ---------------------------------------------------------------------------

class TestBaseAgentInit:
    def test_custom_agent_id(self):
        agent = _ConcreteAgent(agent_id="my-agent-42")
        assert agent.agent_id == "my-agent-42"

    def test_auto_generated_agent_id(self):
        agent = _ConcreteAgent()
        assert agent.agent_id.startswith("testing-")
        assert len(agent.agent_id) > len("testing-")

    def test_initial_status_is_initializing(self):
        agent = _ConcreteAgent()
        assert agent.status == AgentStatus.INITIALIZING

    def test_metrics_start_at_zero(self):
        agent = _ConcreteAgent()
        assert agent.tasks_completed == 0
        assert agent.tasks_failed == 0
        assert agent.total_processing_time == 0.0

    def test_capabilities_stored(self):
        agent = _ConcreteAgent()
        assert "unit_test_generation" in agent.capabilities

    def test_default_port(self):
        agent = _ConcreteAgent()
        assert agent.port == 8080


class TestHealthCheck:
    def setup_method(self):
        self.agent = _ConcreteAgent(agent_id="health-test")
        self.agent.status = AgentStatus.IDLE

    def test_returns_healthy(self):
        result = self.agent.health_check()
        assert result["status"] == "healthy"

    def test_contains_agent_id(self):
        result = self.agent.health_check()
        assert result["agent_id"] == "health-test"

    def test_contains_agent_type(self):
        result = self.agent.health_check()
        assert result["agent_type"] == AgentType.TESTING.value

    def test_uptime_non_negative(self):
        result = self.agent.health_check()
        assert result["uptime_seconds"] >= 0

    def test_success_rate_initially_100(self):
        result = self.agent.health_check()
        assert result["success_rate"] == 100.0

    def test_timestamp_present(self):
        result = self.agent.health_check()
        assert "timestamp" in result
        # Should be parseable ISO datetime
        datetime.fromisoformat(result["timestamp"])


class TestReadinessCheck:
    def setup_method(self):
        self.agent = _ConcreteAgent(agent_id="ready-test")

    def test_not_ready_when_initializing(self):
        self.agent.status = AgentStatus.INITIALIZING
        result = self.agent.readiness_check()
        assert result["ready"] is False
        assert result["can_accept_tasks"] is False

    def test_ready_when_idle(self):
        self.agent.status = AgentStatus.IDLE
        result = self.agent.readiness_check()
        assert result["ready"] is True
        assert result["can_accept_tasks"] is True

    def test_ready_when_working(self):
        self.agent.status = AgentStatus.WORKING
        result = self.agent.readiness_check()
        assert result["ready"] is True
        assert result["can_accept_tasks"] is False

    def test_not_ready_when_error(self):
        self.agent.status = AgentStatus.ERROR
        result = self.agent.readiness_check()
        assert result["ready"] is False


class TestSuccessRate:
    def setup_method(self):
        self.agent = _ConcreteAgent()

    def test_100_when_no_tasks(self):
        assert self.agent.get_success_rate() == 100.0

    def test_100_when_all_succeed(self):
        self.agent.tasks_completed = 5
        assert self.agent.get_success_rate() == 100.0

    def test_0_when_all_fail(self):
        self.agent.tasks_failed = 4
        assert self.agent.get_success_rate() == 0.0

    def test_50_percent(self):
        self.agent.tasks_completed = 5
        self.agent.tasks_failed = 5
        assert self.agent.get_success_rate() == 50.0

    def test_partial_success(self):
        self.agent.tasks_completed = 3
        self.agent.tasks_failed = 1
        assert self.agent.get_success_rate() == 75.0


class TestProcessTask:
    def setup_method(self):
        self.agent = _ConcreteAgent(agent_id="task-test")
        self.agent.status = AgentStatus.IDLE

    def test_successful_task(self):
        result = self.agent.process_task({"task_id": "t1", "action": "echo", "payload": "hello"})
        assert result["success"] is True
        assert result["agent_id"] == "task-test"
        assert result["task_id"] == "t1"
        assert "processing_time_s" in result

    def test_successful_task_increments_counter(self):
        self.agent.process_task({"task_id": "t1", "action": "echo", "payload": "x"})
        assert self.agent.tasks_completed == 1
        assert self.agent.tasks_failed == 0

    def test_failed_task(self):
        result = self.agent.process_task({"task_id": "t2", "action": "raise"})
        assert result["success"] is False
        assert "error" in result

    def test_failed_task_increments_counter(self):
        self.agent.process_task({"task_id": "t2", "action": "raise"})
        assert self.agent.tasks_failed == 1
        assert self.agent.tasks_completed == 0

    def test_status_returns_to_idle_after_success(self):
        self.agent.process_task({"task_id": "t1", "action": "echo", "payload": "x"})
        assert self.agent.status == AgentStatus.IDLE

    def test_status_returns_to_idle_after_failure(self):
        self.agent.process_task({"task_id": "t2", "action": "raise"})
        assert self.agent.status == AgentStatus.IDLE

    def test_current_task_cleared_after_processing(self):
        self.agent.process_task({"task_id": "t1", "action": "echo", "payload": "x"})
        assert self.agent.current_task is None


class TestGetMetrics:
    def setup_method(self):
        self.agent = _ConcreteAgent(agent_id="metrics-test")
        self.agent.status = AgentStatus.IDLE

    def test_structure(self):
        m = self.agent.get_metrics()
        assert "agent_id" in m
        assert "agent_type" in m
        assert "metrics" in m
        inner = m["metrics"]
        for key in ["uptime_seconds", "tasks_completed", "tasks_failed",
                    "success_rate_percent", "avg_processing_time_s"]:
            assert key in inner

    def test_zero_avg_time_when_no_tasks(self):
        m = self.agent.get_metrics()
        assert m["metrics"]["avg_processing_time_s"] == 0

    def test_avg_time_computed_after_tasks(self):
        self.agent.tasks_completed = 2
        self.agent.total_processing_time = 4.0
        m = self.agent.get_metrics()
        assert m["metrics"]["avg_processing_time_s"] == pytest.approx(2.0)


class TestExecuteTaskAbstract:
    def test_raises_not_implemented(self):
        # BaseAgent.execute_task raises NotImplementedError when not overridden
        class _Bare(BaseAgent):
            pass

        agent = _Bare(agent_type=AgentType.TESTING, capabilities=[])
        with pytest.raises(NotImplementedError, match="Specialized agents must implement execute_task"):
            agent.execute_task({})


# ---------------------------------------------------------------------------
# QuantumMechanicsAgent
# ---------------------------------------------------------------------------

class TestQuantumMechanicsAgent:
    def setup_method(self):
        self.agent = QuantumMechanicsAgent(agent_id="qm-test")

    def test_hydrogen_energy_ground_state(self):
        task = {"calculation_type": "hydrogen_energy", "n": 1}
        result = self.agent.execute_task(task)
        assert result["energy_eV"] == pytest.approx(-13.6)
        assert result["n"] == 1

    def test_hydrogen_energy_second_level(self):
        task = {"calculation_type": "hydrogen_energy", "n": 2}
        result = self.agent.execute_task(task)
        assert result["energy_eV"] == pytest.approx(-13.6 / 4)

    def test_uncertainty_principle(self):
        delta_x = 1e-10
        task = {"calculation_type": "uncertainty", "delta_x": delta_x}
        result = self.agent.execute_task(task)
        h_bar = 1.054571817e-34
        expected = h_bar / (2 * delta_x)
        assert result["delta_p_min"] == pytest.approx(expected)

    def test_unknown_calculation_raises(self):
        with pytest.raises(ValueError, match="Unknown calculation type"):
            self.agent.execute_task({"calculation_type": "invalid_calc"})

    def test_capabilities(self):
        assert "hydrogen_energy" in self.agent.capabilities
        assert "uncertainty" in self.agent.capabilities

    def test_agent_type(self):
        assert self.agent.agent_type == AgentType.QUANTUM_MECHANICS


# ---------------------------------------------------------------------------
# CodeReviewAgent
# ---------------------------------------------------------------------------

class TestCodeReviewAgent:
    def setup_method(self):
        self.agent = CodeReviewAgent(agent_id="cr-test")

    def test_syntax_check_clean_code(self):
        task = {"review_type": "syntax_check", "code": "x = 1\ny = 2\n", "language": "python"}
        result = self.agent.execute_task(task)
        assert result["valid"] is True
        assert result["issues"] == []

    def test_complexity_analysis_simple(self):
        task = {"review_type": "complexity_analysis", "code": "def foo():\n    return 1\n"}
        result = self.agent.execute_task(task)
        assert result["cyclomatic_complexity"] >= 1
        assert result["rating"] in ("low", "medium", "high")

    def test_complexity_analysis_complex_code(self):
        code = "\n".join(["if x:" for _ in range(10)])
        task = {"review_type": "complexity_analysis", "code": code}
        result = self.agent.execute_task(task)
        assert result["rating"] in ("medium", "high")

    def test_security_scan_clean(self):
        task = {"review_type": "security_scan", "code": "x = 1 + 2"}
        result = self.agent.execute_task(task)
        assert result["secure"] is True
        assert result["issues"] == []

    def test_security_scan_detects_eval(self):
        task = {"review_type": "security_scan", "code": "result = eval(user_input)"}
        result = self.agent.execute_task(task)
        assert result["secure"] is False
        patterns = [i["pattern"] for i in result["issues"]]
        assert "eval(" in patterns

    def test_security_scan_detects_pickle(self):
        task = {"review_type": "security_scan", "code": "data = pickle.loads(raw)"}
        result = self.agent.execute_task(task)
        assert result["secure"] is False

    def test_style_analysis_clean(self):
        task = {"review_type": "style_analysis", "code": "x = 1\ny = 2"}
        result = self.agent.execute_task(task)
        assert result["style_compliant"] is True

    def test_style_analysis_trailing_whitespace(self):
        task = {"review_type": "style_analysis", "code": "x = 1  \ny = 2"}
        result = self.agent.execute_task(task)
        assert result["style_compliant"] is False
        assert any("trailing whitespace" in i["issue"].lower() for i in result["issues"])

    def test_style_analysis_long_line(self):
        long_line = "x = " + "a" * 120
        task = {"review_type": "style_analysis", "code": long_line}
        result = self.agent.execute_task(task)
        assert result["style_compliant"] is False

    def test_unknown_review_type_raises(self):
        with pytest.raises(ValueError, match="Unknown review type"):
            self.agent.execute_task({"review_type": "magic_review"})


# ---------------------------------------------------------------------------
# TestingAgent
# ---------------------------------------------------------------------------

class TestTestingAgent:
    def setup_method(self):
        self.agent = TestingAgent(agent_id="ta-test")

    def test_unit_test_generation(self):
        task = {"test_type": "unit_test_generation", "function_name": "add", "signature": "add(a, b)"}
        result = self.agent.execute_task(task)
        assert result["function"] == "add"
        assert result["count"] == 3
        names = [t["name"] for t in result["generated_tests"]]
        assert "test_add_basic" in names

    def test_load_test(self):
        task = {"test_type": "load_test", "target_url": "http://example.com", "requests": 200, "concurrent": 20}
        result = self.agent.execute_task(task)
        assert result["total_requests"] == 200
        assert result["concurrent_users"] == 20
        assert "avg_response_time_ms" in result
        assert "error_rate_percent" in result

    def test_coverage_analysis(self):
        task = {"test_type": "coverage_analysis", "module": "base_agent"}
        result = self.agent.execute_task(task)
        assert "line_coverage_percent" in result
        assert "function_coverage_percent" in result
        assert result["module"] == "base_agent"

    def test_test_execution(self):
        task = {"test_type": "test_execution", "test_suite": "tests/"}
        result = self.agent.execute_task(task)
        assert "passed" in result
        assert "failed" in result
        assert result["total_tests"] == result["passed"] + result["failed"] + result["skipped"]

    def test_unknown_test_type_raises(self):
        with pytest.raises(ValueError, match="Unknown test type"):
            self.agent.execute_task({"test_type": "nonexistent"})


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class TestResearchAgent:
    def setup_method(self):
        self.agent = ResearchAgent(agent_id="ra-test")

    def test_literature_search(self):
        task = {"research_type": "literature_search", "query": "machine learning", "max_results": 3}
        result = self.agent.execute_task(task)
        assert result["query"] == "machine learning"
        assert result["results_count"] == 3
        assert len(result["results"]) <= 3

    def test_paper_summary(self):
        task = {"research_type": "paper_summary", "paper_id": "arxiv-1234"}
        result = self.agent.execute_task(task)
        assert result["paper_id"] == "arxiv-1234"
        assert "summary" in result
        assert "key_contributions" in result

    def test_citation_analysis(self):
        task = {"research_type": "citation_analysis", "paper_ids": ["p1", "p2", "p3"]}
        result = self.agent.execute_task(task)
        assert result["papers_analyzed"] == 3
        assert "h_index" in result

    def test_trend_analysis(self):
        task = {"research_type": "trend_analysis", "topic": "AI safety", "time_range": "3y"}
        result = self.agent.execute_task(task)
        assert result["topic"] == "AI safety"
        assert result["time_range"] == "3y"
        assert result["trend"] in ("increasing", "decreasing", "stable")

    def test_unknown_research_type_raises(self):
        with pytest.raises(ValueError, match="Unknown research type"):
            self.agent.execute_task({"research_type": "crystal_ball"})


# ---------------------------------------------------------------------------
# DocumentationAgent
# ---------------------------------------------------------------------------

class TestDocumentationAgent:
    def setup_method(self):
        self.agent = DocumentationAgent(agent_id="doc-test")

    def test_docstring_generation(self):
        task = {"doc_type": "docstring_generation", "function_name": "my_func", "function_code": "def my_func(): pass"}
        result = self.agent.execute_task(task)
        assert result["function_name"] == "my_func"
        assert "my_func" in result["docstring"]

    def test_api_documentation(self):
        task = {"doc_type": "api_documentation", "endpoints": ["/users", "/orders"]}
        result = self.agent.execute_task(task)
        assert result["endpoints_documented"] == 2
        assert "/users" in result["documentation"]["paths"]

    def test_changelog_update(self):
        task = {"doc_type": "changelog_update", "version": "2.0.0", "changes": ["Added feature X", "Fixed bug Y"]}
        result = self.agent.execute_task(task)
        assert result["version"] == "2.0.0"
        assert "2.0.0" in result["changelog_entry"]
        assert "Added feature X" in result["changelog_entry"]

    def test_readme_generation(self):
        task = {"doc_type": "readme_generation", "project_name": "MyProject", "description": "Does stuff"}
        result = self.agent.execute_task(task)
        assert result["project_name"] == "MyProject"
        assert "MyProject" in result["readme"]
        assert "Does stuff" in result["readme"]

    def test_unknown_doc_type_raises(self):
        with pytest.raises(ValueError, match="Unknown documentation type"):
            self.agent.execute_task({"doc_type": "telepathy_docs"})


# ---------------------------------------------------------------------------
# MonitoringAgent
# ---------------------------------------------------------------------------

class TestMonitoringAgent:
    def setup_method(self):
        self.agent = MonitoringAgent(agent_id="mon-test")

    def test_health_monitoring(self):
        task = {"monitor_type": "health_monitoring", "targets": ["svc-a", "svc-b", "svc-c"]}
        result = self.agent.execute_task(task)
        assert result["targets_checked"] == 3
        assert len(result["results"]) == 3

    def test_performance_metrics(self):
        task = {"monitor_type": "performance_metrics", "service": "api-gateway", "time_range": "1h"}
        result = self.agent.execute_task(task)
        assert result["service"] == "api-gateway"
        assert "cpu_usage_percent" in result["metrics"]

    def test_log_analysis(self):
        task = {"monitor_type": "log_analysis", "log_source": "/var/log/app.log", "pattern": "ERROR"}
        result = self.agent.execute_task(task)
        assert result["log_source"] == "/var/log/app.log"
        assert result["pattern"] == "ERROR"
        assert "matches_count" in result

    def test_anomaly_detection(self):
        task = {"monitor_type": "anomaly_detection", "metric": "cpu_usage", "threshold": 2.5}
        result = self.agent.execute_task(task)
        assert result["metric"] == "cpu_usage"
        assert "anomalies_detected" in result

    def test_alert_management(self):
        task = {"monitor_type": "alert_management", "action": "list"}
        result = self.agent.execute_task(task)
        assert result["action"] == "list"
        assert "active_alerts" in result

    def test_unknown_monitor_type_raises(self):
        with pytest.raises(ValueError, match="Unknown monitor type"):
            self.agent.execute_task({"monitor_type": "psychic_monitoring"})


# ---------------------------------------------------------------------------
# IntegrationAgent
# ---------------------------------------------------------------------------

class TestIntegrationAgent:
    def setup_method(self):
        self.agent = IntegrationAgent(agent_id="int-test")

    def test_api_connector(self):
        task = {"integration_type": "api_connector", "source": "salesforce", "destination": "hubspot"}
        result = self.agent.execute_task(task)
        assert result["source"] == "salesforce"
        assert result["destination"] == "hubspot"
        assert result["connection_status"] == "established"

    def test_data_transformation(self):
        task = {"integration_type": "data_transformation", "source_format": "xml", "target_format": "json", "record_count": 500}
        result = self.agent.execute_task(task)
        assert result["source_format"] == "xml"
        assert result["target_format"] == "json"
        assert result["records_processed"] == 500

    def test_webhook_management(self):
        task = {"integration_type": "webhook_management", "action": "register", "url": "https://example.com/hook", "events": ["push", "pr"]}
        result = self.agent.execute_task(task)
        assert result["action"] == "register"
        assert result["webhook_url"] == "https://example.com/hook"
        assert result["status"] == "active"
        assert result["webhook_id"].startswith("wh_")

    def test_sync_orchestration(self):
        task = {"integration_type": "sync_orchestration", "systems": ["crm", "erp"]}
        result = self.agent.execute_task(task)
        assert result["systems"] == ["crm", "erp"]
        assert result["sync_status"] == "completed"

    def test_schema_validation(self):
        task = {"integration_type": "schema_validation", "schema": {"name": "string"}, "data": {"name": "Alice"}}
        result = self.agent.execute_task(task)
        assert result["valid"] is True

    def test_unknown_integration_type_raises(self):
        with pytest.raises(ValueError, match="Unknown integration type"):
            self.agent.execute_task({"integration_type": "magic_pipe"})


# ---------------------------------------------------------------------------
# AnalyticsAgent
# ---------------------------------------------------------------------------

class TestAnalyticsAgent:
    def setup_method(self):
        self.agent = AnalyticsAgent(agent_id="ana-test")

    def test_metrics_aggregation(self):
        task = {"analytics_type": "metrics_aggregation", "metrics": ["requests", "errors"], "time_range": "24h"}
        result = self.agent.execute_task(task)
        assert result["time_range"] == "24h"
        agg = result["aggregations"]
        for key in ("sum", "avg", "min", "max", "p50", "p95", "p99"):
            assert key in agg

    def test_trend_prediction(self):
        task = {"analytics_type": "trend_prediction", "metric": "revenue", "forecast_periods": 5}
        result = self.agent.execute_task(task)
        assert result["metric"] == "revenue"
        assert len(result["predictions"]) == 5
        assert result["model"] == "arima"

    def test_anomaly_scoring(self):
        task = {"analytics_type": "anomaly_scoring", "data_points": list(range(30))}
        result = self.agent.execute_task(task)
        assert result["data_points_analyzed"] == 30
        assert "anomalies_found" in result

    def test_cohort_analysis(self):
        task = {"analytics_type": "cohort_analysis", "cohort_type": "weekly", "metric": "retention"}
        result = self.agent.execute_task(task)
        assert result["cohort_type"] == "weekly"
        assert len(result["cohorts"]) > 0

    def test_funnel_analysis(self):
        steps = ["visit", "signup", "activate", "purchase"]
        task = {"analytics_type": "funnel_analysis", "funnel_steps": steps}
        result = self.agent.execute_task(task)
        assert result["funnel_steps"] == steps
        assert len(result["conversion_rates"]) == len(steps)
        assert "overall_conversion" in result

    def test_unknown_analytics_type_raises(self):
        with pytest.raises(ValueError, match="Unknown analytics type"):
            self.agent.execute_task({"analytics_type": "voodoo_stats"})


# ---------------------------------------------------------------------------
# DeploymentAgent
# ---------------------------------------------------------------------------

class TestDeploymentAgent:
    def setup_method(self):
        self.agent = DeploymentAgent(agent_id="dep-test")

    def test_build_execution(self):
        task = {"deploy_type": "build_execution", "project": "my-app", "branch": "main"}
        result = self.agent.execute_task(task)
        assert result["project"] == "my-app"
        assert result["branch"] == "main"
        assert result["status"] == "success"
        assert result["build_id"].startswith("build-")

    def test_deployment_orchestration(self):
        task = {"deploy_type": "deployment_orchestration", "environment": "production", "version": "v3.1.0"}
        result = self.agent.execute_task(task)
        assert result["environment"] == "production"
        assert result["version"] == "v3.1.0"
        assert result["status"] == "completed"
        assert result["deployment_id"].startswith("deploy-")

    def test_rollback_management(self):
        task = {"deploy_type": "rollback_management", "environment": "staging", "target_version": "v2.0.0"}
        result = self.agent.execute_task(task)
        assert result["rolled_back_to"] == "v2.0.0"
        assert result["status"] == "completed"

    def test_canary_deployment(self):
        task = {"deploy_type": "canary_deployment", "environment": "production", "version": "v3.2.0", "traffic_percent": 5}
        result = self.agent.execute_task(task)
        assert result["canary_traffic_percent"] == 5
        assert result["status"] == "active"
        assert "health_metrics" in result

    def test_environment_provisioning(self):
        task = {
            "deploy_type": "environment_provisioning",
            "environment_name": "staging-qa",
            "config": {"instances": 2, "cpu": 2, "memory": 4},
        }
        result = self.agent.execute_task(task)
        assert result["environment_name"] == "staging-qa"
        assert result["status"] == "provisioned"
        assert result["resources"]["instances"] == 2
        assert "staging-qa" in result["endpoints"]["api"]

    def test_unknown_deploy_type_raises(self):
        with pytest.raises(ValueError, match="Unknown deployment type"):
            self.agent.execute_task({"deploy_type": "warp_drive"})


# ---------------------------------------------------------------------------
# AGENT_REGISTRY and create_agent factory
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    def test_all_expected_types_registered(self):
        expected_keys = {
            "quantum_mechanics", "code_review", "testing", "research",
            "documentation", "monitoring", "integration", "analytics", "deployment",
        }
        assert expected_keys.issubset(set(AGENT_REGISTRY.keys()))

    def test_registry_maps_to_correct_classes(self):
        assert AGENT_REGISTRY["quantum_mechanics"] is QuantumMechanicsAgent
        assert AGENT_REGISTRY["code_review"] is CodeReviewAgent
        assert AGENT_REGISTRY["deployment"] is DeploymentAgent


class TestCreateAgent:
    def test_creates_quantum_mechanics_agent(self):
        agent = create_agent("quantum_mechanics")
        assert isinstance(agent, QuantumMechanicsAgent)

    def test_creates_code_review_agent(self):
        agent = create_agent("code_review")
        assert isinstance(agent, CodeReviewAgent)

    def test_creates_testing_agent(self):
        agent = create_agent("testing")
        assert isinstance(agent, TestingAgent)

    def test_creates_research_agent(self):
        agent = create_agent("research")
        assert isinstance(agent, ResearchAgent)

    def test_creates_documentation_agent(self):
        agent = create_agent("documentation")
        assert isinstance(agent, DocumentationAgent)

    def test_creates_monitoring_agent(self):
        agent = create_agent("monitoring")
        assert isinstance(agent, MonitoringAgent)

    def test_creates_integration_agent(self):
        agent = create_agent("integration")
        assert isinstance(agent, IntegrationAgent)

    def test_creates_analytics_agent(self):
        agent = create_agent("analytics")
        assert isinstance(agent, AnalyticsAgent)

    def test_creates_deployment_agent(self):
        agent = create_agent("deployment")
        assert isinstance(agent, DeploymentAgent)

    def test_custom_agent_id_passed_through(self):
        agent = create_agent("testing", agent_id="custom-id-99")
        assert agent.agent_id == "custom-id-99"

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("does_not_exist")

    def test_error_message_lists_available_types(self):
        with pytest.raises(ValueError) as exc_info:
            create_agent("bad_type")
        assert "quantum_mechanics" in str(exc_info.value)
