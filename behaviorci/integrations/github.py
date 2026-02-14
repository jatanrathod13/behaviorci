"""GitHub integration for posting PR comments."""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, NamedTuple

from behaviorci.diff.models import DiffResult


class PRInfo(NamedTuple):
    """Information about the current PR."""
    owner: str
    repo: str
    number: int
    sha: str


class GitHubReporter:
    """Post behavior test results as GitHub PR comments.

    Supports two methods:
    1. gh CLI (preferred if available)
    2. GitHub REST API with token
    """

    def __init__(self, token: str | None = None, use_gh_cli: bool = True):
        """Initialize GitHub reporter.

        Args:
            token: GitHub token for API auth (optional if using gh CLI)
            use_gh_cli: Prefer gh CLI over REST API (default: True)
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.use_gh_cli = use_gh_cli and self._gh_cli_available()

    def _gh_cli_available(self) -> bool:
        """Check if gh CLI is available."""
        try:
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def detect_pr_info() -> PRInfo | None:
        """Detect PR info from CI environment variables.

        Supports:
        - GitHub Actions (GITHUB_REPOSITORY, GITHUB_PR_NUMBER, GITHUB_SHA)
        - Manual environment overrides

        Returns:
            PRInfo if running in PR context, None otherwise
        """
        # GitHub Actions environment
        repo = os.environ.get("GITHUB_REPOSITORY")
        pr_number = os.environ.get("GITHUB_PR_NUMBER") or os.environ.get("GITHUB_EVENT_PULL_REQUEST", {}).get("number")
        sha = os.environ.get("GITHUB_SHA")

        # Try to parse from GITHUB_PR_NUMBER if it's a URL
        if pr_number and isinstance(pr_number, str) and "/" in pr_number:
            # Extract PR number from URL like https://github.com/owner/repo/pull/123
            match = re.search(r"/pull/(\d+)", str(pr_number))
            if match:
                pr_number = match.group(1)

        # Parse owner/repo from GITHUB_REPOSITORY
        if repo and "/" in repo:
            owner, repo_name = repo.split("/", 1)
        else:
            return None

        # Get PR number from GitHub event if not set
        if not pr_number:
            event_path = os.environ.get("GITHUB_EVENT_PATH")
            if event_path and os.path.exists(event_path):
                try:
                    with open(event_path) as f:
                        event = json.load(f)
                        if isinstance(event, dict):
                            pr_number = event.get("pull_request", {}).get("number")
                            sha = sha or event.get("pull_request", {}).get("head", {}).get("sha")
                except (json.JSONDecodeError, IOError):
                    pass

        if not pr_number or not owner or not repo_name:
            return None

        return PRInfo(
            owner=owner,
            repo=repo_name,
            number=int(pr_number),
            sha=sha or "unknown",
        )

    @staticmethod
    def is_ci_environment() -> bool:
        """Check if running in a CI environment."""
        return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))

    def _build_pr_url(self, pr_info: PRInfo) -> str:
        """Build the PR URL from PR info."""
        return f"https://github.com/{pr_info.owner}/{pr_info.repo}/pull/{pr_info.number}"

    def _post_via_gh_cli(self, pr_info: PRInfo, body: str) -> bool:
        """Post comment using gh CLI."""
        try:
            result = subprocess.run(
                [
                    "gh", "pr", "comment",
                    str(pr_info.number),
                    "--repo", f"{pr_info.owner}/{pr_info.repo}",
                    "--body", body,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "GH_TOKEN": self.token} if self.token else None,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _post_via_api(self, pr_info: PRInfo, body: str) -> bool:
        """Post comment using GitHub REST API."""
        try:
            import requests
        except ImportError:
            return False

        if not self.token:
            return False

        url = f"https://api.github.com/repos/{pr_info.owner}/{pr_info.repo}/issues/{pr_info.number}/comments"

        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"token {self.token}",
                    "Accept": "application/vnd.github.v3+json",
                    "Content-Type": "application/json",
                },
                json={"body": body},
                timeout=30,
            )
            return response.status_code in (200, 201)
        except Exception:
            return False

    def post_diff_comment(
        self,
        pr_url: str | None,
        diff_result: DiffResult,
        markdown_report: str,
    ) -> bool:
        """Post a comment to a GitHub PR with the behavior test results.

        Args:
            pr_url: PR URL (auto-detected if not provided)
            diff_result: The diff result from behavior tests
            markdown_report: Formatted markdown report to post

        Returns:
            True if comment was posted successfully
        """
        # Try to get PR info
        pr_info = None

        if pr_url:
            pr_info = self._parse_pr_url(pr_url)
        else:
            pr_info = self.detect_pr_info()

        if not pr_info:
            return False

        # Build comment body
        body = self._format_comment(diff_result, markdown_report)

        # Try posting via gh CLI first
        if self.use_gh_cli:
            if self._post_via_gh_cli(pr_info, body):
                return True
            # Fall through to API if gh CLI fails

        # Fall back to REST API
        return self._post_via_api(pr_info, body)

    def _parse_pr_url(self, url: str) -> PRInfo | None:
        """Parse PR URL to extract owner, repo, and PR number."""
        # Match patterns like:
        # https://github.com/owner/repo/pull/123
        # github.com/owner/repo/pull/123
        match = re.match(r"github\.com[/:]([^/]+)/([^/]+)/pull/(\d+)", url)
        if match:
            return PRInfo(
                owner=match.group(1),
                repo=match.group(2),
                number=int(match.group(3)),
                sha="unknown",
            )
        return None

    def _format_comment(self, diff_result: DiffResult, report: str) -> str:
        """Format the comment body with results."""
        # Determine status emoji and text
        if diff_result.has_regressions():
            status_emoji = "🔴"
            status_text = "Behavior Regression Detected"
        elif diff_result.has_improvements():
            status_emoji = "🟢"
            status_text = "Behavior Improved"
        else:
            status_emoji = "✅"
            status_text = "No Changes"

        # Calculate counts from entries
        total = len(diff_result.entries)
        new_failures = len(diff_result.new_failures)
        fixed = len(diff_result.fixed_cases)
        regressions = len(diff_result.regressions)

        body = f"""## Behavior Test Results {status_emoji}

**{status_text}**

| Metric | Value |
|--------|-------|
| Total Cases | {total} |
| New Failures | {new_failures} |
| Fixed Cases | {fixed} |
| Regressions | {regressions} |

---
{report}
"""
        return body
