"""
Parallel Orchestrator
=====================

Coordinates parallel execution of independent features using multiple agent processes.
Uses dependency-aware scheduling to ensure features are only started when their
dependencies are satisfied.

Usage:
    python parallel_orchestrator.py --project-dir my-app --max-concurrency 3
"""

import asyncio
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Awaitable

import psutil

from api.database import Feature, create_database
from api.dependency_resolver import are_dependencies_satisfied, compute_scheduling_scores

# Root directory of autocoder (where this script and autonomous_agent_demo.py live)
AUTOCODER_ROOT = Path(__file__).parent.resolve()

# Performance: Limit parallel agents to prevent memory exhaustion
MAX_PARALLEL_AGENTS = 5
DEFAULT_CONCURRENCY = 3
POLL_INTERVAL = 5  # seconds between checking for ready features
MAX_FEATURE_RETRIES = 3  # Maximum times to retry a failed feature


def _kill_process_tree(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """Kill a process and all its child processes.

    On Windows, subprocess.terminate() only kills the immediate process, leaving
    orphaned child processes (e.g., spawned browser instances). This function
    uses psutil to kill the entire process tree.

    Args:
        proc: The subprocess.Popen object to kill
        timeout: Seconds to wait for graceful termination before force-killing
    """
    try:
        parent = psutil.Process(proc.pid)
        # Get all children recursively before terminating
        children = parent.children(recursive=True)

        # Terminate children first (graceful)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Wait for children to terminate
        _, still_alive = psutil.wait_procs(children, timeout=timeout)

        # Force kill any remaining children
        for child in still_alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        # Now terminate the parent
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    except psutil.NoSuchProcess:
        # Process already dead, just ensure cleanup
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except (subprocess.TimeoutExpired, OSError):
            try:
                proc.kill()
            except OSError:
                pass


class ParallelOrchestrator:
    """Orchestrates parallel execution of independent features."""

    def __init__(
        self,
        project_dir: Path,
        max_concurrency: int = DEFAULT_CONCURRENCY,
        model: str = None,
        yolo_mode: bool = False,
        on_output: Callable[[int, str], None] = None,
        on_status: Callable[[int, str], None] = None,
    ):
        """Initialize the orchestrator.

        Args:
            project_dir: Path to the project directory
            max_concurrency: Maximum number of concurrent agents (1-5)
            model: Claude model to use (or None for default)
            yolo_mode: Whether to run in YOLO mode (skip browser testing)
            on_output: Callback for agent output (feature_id, line)
            on_status: Callback for agent status changes (feature_id, status)
        """
        self.project_dir = project_dir
        self.max_concurrency = min(max(max_concurrency, 1), MAX_PARALLEL_AGENTS)
        self.model = model
        self.yolo_mode = yolo_mode
        self.on_output = on_output
        self.on_status = on_status

        # Thread-safe state
        self._lock = threading.Lock()
        self.running_agents: dict[int, subprocess.Popen] = {}
        self.abort_events: dict[int, threading.Event] = {}
        self.is_running = False

        # Track feature failures to prevent infinite retry loops
        self._failure_counts: dict[int, int] = {}

        # Database session for this orchestrator
        self._engine, self._session_maker = create_database(project_dir)

    def get_session(self):
        """Get a new database session."""
        return self._session_maker()

    def get_resumable_features(self) -> list[dict]:
        """Get features that were left in_progress from a previous session.

        These are features where in_progress=True but passes=False, and they're
        not currently being worked on by this orchestrator. This handles the case
        where a previous session was interrupted before completing the feature.
        """
        session = self.get_session()
        try:
            # Find features that are in_progress but not complete
            stale = session.query(Feature).filter(
                Feature.in_progress == True,
                Feature.passes == False
            ).all()

            resumable = []
            for f in stale:
                # Skip if already running in this orchestrator instance
                with self._lock:
                    if f.id in self.running_agents:
                        continue
                # Skip if feature has failed too many times
                if self._failure_counts.get(f.id, 0) >= MAX_FEATURE_RETRIES:
                    continue
                resumable.append(f.to_dict())

            # Sort by scheduling score (higher = first), then priority, then id
            all_dicts = [f.to_dict() for f in session.query(Feature).all()]
            scores = compute_scheduling_scores(all_dicts)
            resumable.sort(key=lambda f: (-scores.get(f["id"], 0), f["priority"], f["id"]))
            return resumable
        finally:
            session.close()

    def get_ready_features(self) -> list[dict]:
        """Get features with satisfied dependencies, not already running."""
        session = self.get_session()
        try:
            all_features = session.query(Feature).all()
            all_dicts = [f.to_dict() for f in all_features]

            ready = []
            for f in all_features:
                if f.passes or f.in_progress:
                    continue
                # Skip if already running in this orchestrator
                with self._lock:
                    if f.id in self.running_agents:
                        continue
                # Skip if feature has failed too many times
                if self._failure_counts.get(f.id, 0) >= MAX_FEATURE_RETRIES:
                    continue
                # Check dependencies
                if are_dependencies_satisfied(f.to_dict(), all_dicts):
                    ready.append(f.to_dict())

            # Sort by scheduling score (higher = first), then priority, then id
            scores = compute_scheduling_scores(all_dicts)
            ready.sort(key=lambda f: (-scores.get(f["id"], 0), f["priority"], f["id"]))
            return ready
        finally:
            session.close()

    def get_all_complete(self) -> bool:
        """Check if all features are complete or permanently failed."""
        session = self.get_session()
        try:
            all_features = session.query(Feature).all()
            for f in all_features:
                if f.passes:
                    continue  # Completed successfully
                if self._failure_counts.get(f.id, 0) >= MAX_FEATURE_RETRIES:
                    continue  # Permanently failed, count as "done"
                return False  # Still workable
            return True
        finally:
            session.close()

    def start_feature(self, feature_id: int, resume: bool = False) -> tuple[bool, str]:
        """Start a single feature agent.

        Args:
            feature_id: ID of the feature to start
            resume: If True, resume a feature that's already in_progress from a previous session

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if feature_id in self.running_agents:
                return False, "Feature already running"
            if len(self.running_agents) >= self.max_concurrency:
                return False, "At max concurrency"

        # Mark as in_progress in database (or verify it's resumable)
        session = self.get_session()
        try:
            feature = session.query(Feature).filter(Feature.id == feature_id).first()
            if not feature:
                return False, "Feature not found"
            if feature.passes:
                return False, "Feature already complete"

            if resume:
                # Resuming: feature should already be in_progress
                if not feature.in_progress:
                    return False, "Feature not in progress, cannot resume"
            else:
                # Starting fresh: feature should not be in_progress
                if feature.in_progress:
                    return False, "Feature already in progress"
                feature.in_progress = True
                session.commit()
        finally:
            session.close()

        # Create abort event
        abort_event = threading.Event()

        # Start subprocess for this feature
        cmd = [
            sys.executable,
            "-u",  # Force unbuffered stdout/stderr
            str(AUTOCODER_ROOT / "autonomous_agent_demo.py"),
            "--project-dir", str(self.project_dir),
            "--max-iterations", "1",  # Single feature mode
            "--feature-id", str(feature_id),  # Work on this specific feature only
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.yolo_mode:
            cmd.append("--yolo")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(AUTOCODER_ROOT),  # Run from autocoder root for proper imports
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception as e:
            # Reset in_progress on failure
            session = self.get_session()
            try:
                feature = session.query(Feature).filter(Feature.id == feature_id).first()
                if feature:
                    feature.in_progress = False
                    session.commit()
            finally:
                session.close()
            return False, f"Failed to start agent: {e}"

        with self._lock:
            self.running_agents[feature_id] = proc
            self.abort_events[feature_id] = abort_event

        # Start output reader thread
        threading.Thread(
            target=self._read_output,
            args=(feature_id, proc, abort_event),
            daemon=True
        ).start()

        if self.on_status:
            self.on_status(feature_id, "running")

        print(f"Started agent for feature #{feature_id}", flush=True)
        return True, f"Started feature {feature_id}"

    def _read_output(self, feature_id: int, proc: subprocess.Popen, abort: threading.Event):
        """Read output from subprocess and emit events."""
        try:
            for line in proc.stdout:
                if abort.is_set():
                    break
                line = line.rstrip()
                if self.on_output:
                    self.on_output(feature_id, line)
                else:
                    print(f"[Feature #{feature_id}] {line}", flush=True)
            proc.wait()
        finally:
            self._on_feature_complete(feature_id, proc.returncode)

    def _on_feature_complete(self, feature_id: int, return_code: int):
        """Handle feature completion.

        ALWAYS clears in_progress when agent exits, regardless of success/failure.
        This prevents features from getting stuck if an agent crashes or is killed.
        The agent marks features as passing BEFORE clearing in_progress, so this
        is safe - we won't accidentally clear a feature that's being worked on.
        """
        with self._lock:
            self.running_agents.pop(feature_id, None)
            self.abort_events.pop(feature_id, None)

        # ALWAYS clear in_progress when agent exits to prevent stuck features
        # The agent marks features as passing before clearing in_progress,
        # so if in_progress is still True here, the feature didn't complete successfully
        session = self.get_session()
        try:
            feature = session.query(Feature).filter(Feature.id == feature_id).first()
            if feature and feature.in_progress and not feature.passes:
                feature.in_progress = False
                session.commit()
        finally:
            session.close()

        # Track failures to prevent infinite retry loops
        if return_code != 0:
            with self._lock:
                self._failure_counts[feature_id] = self._failure_counts.get(feature_id, 0) + 1
                failure_count = self._failure_counts[feature_id]
            if failure_count >= MAX_FEATURE_RETRIES:
                print(f"Feature #{feature_id} has failed {failure_count} times, will not retry", flush=True)

        status = "completed" if return_code == 0 else "failed"
        if self.on_status:
            self.on_status(feature_id, status)
        # CRITICAL: This print triggers the WebSocket to emit agent_update with state='error' or 'success'
        print(f"Feature #{feature_id} {status}", flush=True)

    def stop_feature(self, feature_id: int) -> tuple[bool, str]:
        """Stop a running feature agent and all its child processes."""
        with self._lock:
            if feature_id not in self.running_agents:
                return False, "Feature not running"

            abort = self.abort_events.get(feature_id)
            proc = self.running_agents.get(feature_id)

        if abort:
            abort.set()
        if proc:
            # Kill entire process tree to avoid orphaned children (e.g., browser instances)
            _kill_process_tree(proc, timeout=5.0)

        return True, f"Stopped feature {feature_id}"

    def stop_all(self) -> None:
        """Stop all running feature agents."""
        self.is_running = False
        with self._lock:
            feature_ids = list(self.running_agents.keys())

        for fid in feature_ids:
            self.stop_feature(fid)

    async def run_loop(self):
        """Main orchestration loop."""
        self.is_running = True

        print(f"Starting parallel orchestrator with max_concurrency={self.max_concurrency}", flush=True)
        print(f"Project: {self.project_dir}", flush=True)
        print(flush=True)

        # Check for features to resume from previous session
        resumable = self.get_resumable_features()
        if resumable:
            print(f"Found {len(resumable)} feature(s) to resume from previous session:", flush=True)
            for f in resumable:
                print(f"  - Feature #{f['id']}: {f['name']}", flush=True)
            print(flush=True)

        while self.is_running:
            try:
                # Check if all complete
                if self.get_all_complete():
                    print("\nAll features complete!", flush=True)
                    break

                # Check capacity
                with self._lock:
                    current = len(self.running_agents)
                if current >= self.max_concurrency:
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                # Priority 1: Resume features from previous session
                resumable = self.get_resumable_features()
                if resumable:
                    slots = self.max_concurrency - current
                    for feature in resumable[:slots]:
                        print(f"Resuming feature #{feature['id']}: {feature['name']}", flush=True)
                        self.start_feature(feature["id"], resume=True)
                    await asyncio.sleep(2)
                    continue

                # Priority 2: Start new ready features
                ready = self.get_ready_features()
                if not ready:
                    # Wait for running features to complete
                    if current > 0:
                        await asyncio.sleep(POLL_INTERVAL)
                        continue
                    else:
                        # No ready features and nothing running - might be blocked
                        print("No ready features available. All remaining features may be blocked by dependencies.", flush=True)
                        await asyncio.sleep(POLL_INTERVAL * 2)
                        continue

                # Start features up to capacity
                slots = self.max_concurrency - current
                for feature in ready[:slots]:
                    print(f"Starting feature #{feature['id']}: {feature['name']}", flush=True)
                    self.start_feature(feature["id"])

                await asyncio.sleep(2)  # Brief pause between starts

            except Exception as e:
                print(f"Orchestrator error: {e}", flush=True)
                await asyncio.sleep(POLL_INTERVAL)

        # Wait for remaining agents to complete
        print("Waiting for running agents to complete...", flush=True)
        while True:
            with self._lock:
                if not self.running_agents:
                    break
            await asyncio.sleep(1)

        print("Orchestrator finished.", flush=True)

    def get_status(self) -> dict:
        """Get current orchestrator status."""
        with self._lock:
            return {
                "running_features": list(self.running_agents.keys()),
                "count": len(self.running_agents),
                "max_concurrency": self.max_concurrency,
                "is_running": self.is_running,
            }


async def run_parallel_orchestrator(
    project_dir: Path,
    max_concurrency: int = DEFAULT_CONCURRENCY,
    model: str = None,
    yolo_mode: bool = False,
) -> None:
    """Run the parallel orchestrator.

    Args:
        project_dir: Path to the project directory
        max_concurrency: Maximum number of concurrent agents
        model: Claude model to use
        yolo_mode: Whether to run in YOLO mode
    """
    orchestrator = ParallelOrchestrator(
        project_dir=project_dir,
        max_concurrency=max_concurrency,
        model=model,
        yolo_mode=yolo_mode,
    )

    try:
        await orchestrator.run_loop()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Stopping agents...", flush=True)
        orchestrator.stop_all()


def main():
    """Main entry point for parallel orchestration."""
    import argparse
    from dotenv import load_dotenv
    from registry import DEFAULT_MODEL, get_project_path

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Parallel Feature Orchestrator - Run multiple agent instances",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        required=True,
        help="Project directory path (absolute) or registered project name",
    )
    parser.add_argument(
        "--max-concurrency",
        "-p",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum concurrent agents (1-{MAX_PARALLEL_AGENTS}, default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        default=False,
        help="Enable YOLO mode: rapid prototyping without browser testing",
    )

    args = parser.parse_args()

    # Resolve project directory
    project_dir_input = args.project_dir
    project_dir = Path(project_dir_input)

    if project_dir.is_absolute():
        if not project_dir.exists():
            print(f"Error: Project directory does not exist: {project_dir}", flush=True)
            sys.exit(1)
    else:
        registered_path = get_project_path(project_dir_input)
        if registered_path:
            project_dir = registered_path
        else:
            print(f"Error: Project '{project_dir_input}' not found in registry", flush=True)
            sys.exit(1)

    try:
        asyncio.run(run_parallel_orchestrator(
            project_dir=project_dir,
            max_concurrency=args.max_concurrency,
            model=args.model,
            yolo_mode=args.yolo,
        ))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", flush=True)


if __name__ == "__main__":
    main()
