#!/usr/bin/env python3
"""
BlackRoad 30K Agent Performance Benchmark Tool

Phase 2 Scale Testing utility for measuring agent performance at scale.
Tests latency, throughput, and resource utilization across agent deployments.

Usage:
    python benchmark.py --agents 100 --duration 60
    python benchmark.py --agents 1000 --concurrent 50 --duration 300
"""

import argparse
import asyncio
import aiohttp
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark request."""
    agent_id: str
    success: bool
    latency_ms: float
    status_code: int
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies_ms: list = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    errors: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def requests_per_second(self) -> float:
        if self.duration_seconds == 0:
            return 0.0
        return self.total_requests / self.duration_seconds

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        if len(self.latencies_ms) < 2:
            return self.avg_latency_ms
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    @property
    def p99_latency_ms(self) -> float:
        if len(self.latencies_ms) < 2:
            return self.avg_latency_ms
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]

    def to_dict(self) -> dict:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate_percent": round(self.success_rate, 2),
                "duration_seconds": round(self.duration_seconds, 2),
                "requests_per_second": round(self.requests_per_second, 2),
            },
            "latency_ms": {
                "avg": round(self.avg_latency_ms, 2),
                "p50": round(self.p50_latency_ms, 2),
                "p95": round(self.p95_latency_ms, 2),
                "p99": round(self.p99_latency_ms, 2),
                "min": round(min(self.latencies_ms), 2) if self.latencies_ms else 0,
                "max": round(max(self.latencies_ms), 2) if self.latencies_ms else 0,
            },
            "errors": self.errors[:10],  # First 10 errors
        }


class AgentBenchmark:
    """Performance benchmark tool for BlackRoad agents."""

    def __init__(
        self,
        base_url: str = "http://localhost",
        num_agents: int = 100,
        concurrent_requests: int = 10,
        duration_seconds: int = 60,
        port_start: int = 8080,
    ):
        self.base_url = base_url
        self.num_agents = num_agents
        self.concurrent_requests = concurrent_requests
        self.duration_seconds = duration_seconds
        self.port_start = port_start
        self.report = BenchmarkReport()

    def get_agent_url(self, agent_idx: int) -> str:
        """Generate URL for agent by index."""
        port = self.port_start + agent_idx
        return f"{self.base_url}:{port}"

    async def health_check(
        self, session: aiohttp.ClientSession, agent_idx: int
    ) -> BenchmarkResult:
        """Perform health check on a single agent."""
        url = f"{self.get_agent_url(agent_idx)}/health"
        start_time = time.perf_counter()

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                latency_ms = (time.perf_counter() - start_time) * 1000
                return BenchmarkResult(
                    agent_id=f"agent-{agent_idx}",
                    success=resp.status == 200,
                    latency_ms=latency_ms,
                    status_code=resp.status,
                )
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                agent_id=f"agent-{agent_idx}",
                success=False,
                latency_ms=latency_ms,
                status_code=0,
                error="Timeout",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                agent_id=f"agent-{agent_idx}",
                success=False,
                latency_ms=latency_ms,
                status_code=0,
                error=str(e),
            )

    async def send_task(
        self, session: aiohttp.ClientSession, agent_idx: int, task_data: dict
    ) -> BenchmarkResult:
        """Send a task to a single agent."""
        url = f"{self.get_agent_url(agent_idx)}/task"
        start_time = time.perf_counter()

        try:
            async with session.post(
                url,
                json=task_data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                latency_ms = (time.perf_counter() - start_time) * 1000
                return BenchmarkResult(
                    agent_id=f"agent-{agent_idx}",
                    success=resp.status == 200,
                    latency_ms=latency_ms,
                    status_code=resp.status,
                )
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                agent_id=f"agent-{agent_idx}",
                success=False,
                latency_ms=latency_ms,
                status_code=0,
                error="Timeout",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                agent_id=f"agent-{agent_idx}",
                success=False,
                latency_ms=latency_ms,
                status_code=0,
                error=str(e),
            )

    async def run_health_benchmark(self) -> BenchmarkReport:
        """Run health check benchmark across all agents."""
        logger.info(f"Starting health check benchmark for {self.num_agents} agents")
        logger.info(f"Concurrent requests: {self.concurrent_requests}")
        logger.info(f"Duration: {self.duration_seconds} seconds")

        self.report = BenchmarkReport()
        self.report.start_time = time.time()
        end_time = self.report.start_time + self.duration_seconds

        connector = aiohttp.TCPConnector(limit=self.concurrent_requests)
        async with aiohttp.ClientSession(connector=connector) as session:
            while time.time() < end_time:
                # Create batch of concurrent requests
                tasks = []
                for i in range(self.concurrent_requests):
                    agent_idx = (self.report.total_requests + i) % self.num_agents
                    tasks.append(self.health_check(session, agent_idx))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        self.report.failed_requests += 1
                        self.report.errors.append(str(result))
                    elif isinstance(result, BenchmarkResult):
                        self.report.total_requests += 1
                        if result.success:
                            self.report.successful_requests += 1
                            self.report.latencies_ms.append(result.latency_ms)
                        else:
                            self.report.failed_requests += 1
                            if result.error:
                                self.report.errors.append(result.error)

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)

        self.report.end_time = time.time()
        return self.report

    async def run_task_benchmark(self) -> BenchmarkReport:
        """Run task submission benchmark across all agents."""
        logger.info(f"Starting task benchmark for {self.num_agents} agents")
        logger.info(f"Concurrent requests: {self.concurrent_requests}")
        logger.info(f"Duration: {self.duration_seconds} seconds")

        self.report = BenchmarkReport()
        self.report.start_time = time.time()
        end_time = self.report.start_time + self.duration_seconds

        task_data = {
            "type": "benchmark",
            "payload": {"operation": "ping", "timestamp": time.time()},
        }

        connector = aiohttp.TCPConnector(limit=self.concurrent_requests)
        async with aiohttp.ClientSession(connector=connector) as session:
            while time.time() < end_time:
                tasks = []
                for i in range(self.concurrent_requests):
                    agent_idx = (self.report.total_requests + i) % self.num_agents
                    tasks.append(self.send_task(session, agent_idx, task_data))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        self.report.failed_requests += 1
                        self.report.errors.append(str(result))
                    elif isinstance(result, BenchmarkResult):
                        self.report.total_requests += 1
                        if result.success:
                            self.report.successful_requests += 1
                            self.report.latencies_ms.append(result.latency_ms)
                        else:
                            self.report.failed_requests += 1
                            if result.error:
                                self.report.errors.append(result.error)

                await asyncio.sleep(0.01)

        self.report.end_time = time.time()
        return self.report


def print_report(report: BenchmarkReport) -> None:
    """Print benchmark report in a formatted way."""
    report_dict = report.to_dict()

    print("\n" + "=" * 60)
    print("  BLACKROAD AGENT BENCHMARK REPORT")
    print("=" * 60)
    print(f"  Timestamp: {report_dict['timestamp']}")
    print("-" * 60)

    summary = report_dict["summary"]
    print("\n  SUMMARY")
    print(f"    Total Requests:      {summary['total_requests']:,}")
    print(f"    Successful:          {summary['successful_requests']:,}")
    print(f"    Failed:              {summary['failed_requests']:,}")
    print(f"    Success Rate:        {summary['success_rate_percent']}%")
    print(f"    Duration:            {summary['duration_seconds']}s")
    print(f"    Throughput:          {summary['requests_per_second']:.2f} req/s")

    latency = report_dict["latency_ms"]
    print("\n  LATENCY (ms)")
    print(f"    Average:             {latency['avg']:.2f}")
    print(f"    P50 (Median):        {latency['p50']:.2f}")
    print(f"    P95:                 {latency['p95']:.2f}")
    print(f"    P99:                 {latency['p99']:.2f}")
    print(f"    Min:                 {latency['min']:.2f}")
    print(f"    Max:                 {latency['max']:.2f}")

    if report_dict["errors"]:
        print("\n  ERRORS (first 10)")
        for i, error in enumerate(report_dict["errors"], 1):
            print(f"    {i}. {error[:50]}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="BlackRoad 30K Agent Performance Benchmark Tool"
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=100,
        help="Number of agents to benchmark (default: 100)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost",
        help="Base URL for agents (default: http://localhost)",
    )
    parser.add_argument(
        "--port-start",
        type=int,
        default=8080,
        help="Starting port number (default: 8080)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["health", "task"],
        default="health",
        help="Benchmark type: health checks or task submission (default: health)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON report",
    )

    args = parser.parse_args()

    benchmark = AgentBenchmark(
        base_url=args.base_url,
        num_agents=args.agents,
        concurrent_requests=args.concurrent,
        duration_seconds=args.duration,
        port_start=args.port_start,
    )

    logger.info("=" * 50)
    logger.info("  BLACKROAD 30K AGENT BENCHMARK")
    logger.info("=" * 50)
    logger.info(f"  Agents:     {args.agents}")
    logger.info(f"  Concurrent: {args.concurrent}")
    logger.info(f"  Duration:   {args.duration}s")
    logger.info(f"  Type:       {args.type}")
    logger.info("=" * 50)

    if args.type == "health":
        report = asyncio.run(benchmark.run_health_benchmark())
    else:
        report = asyncio.run(benchmark.run_task_benchmark())

    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Report saved to {args.output}")

    # Return exit code based on success rate
    if report.success_rate >= 98:
        logger.info("Benchmark PASSED (success rate >= 98%)")
        return 0
    else:
        logger.warning(f"Benchmark FAILED (success rate {report.success_rate:.2f}% < 98%)")
        return 1


if __name__ == "__main__":
    exit(main())
