"""Push a local study JSON artifact to GitHub via the Contents API.

Usage:
  python study/push_artifact.py /path/to/study_....json --token $GITHUB_TOKEN
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path


def github_request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "recap-study-push",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {url} failed: {exc.code} {detail}") from exc


def push_file(
    *,
    local_path: Path,
    token: str,
    owner: str,
    repo: str,
    branch: str,
    dest_path: str,
    message: str,
) -> str:
    content_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{dest_path}"

    sha = None
    try:
        existing = github_request("GET", f"{api}?ref={branch}", token)
        sha = existing.get("sha")
    except RuntimeError:
        sha = None

    payload = {
        "message": message,
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    result = github_request("PUT", api, token, payload)
    html_url = result.get("content", {}).get("html_url") or result.get("commit", {}).get("html_url")
    return str(html_url)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("local_path", type=Path)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument("--owner", default="laraib-b")
    parser.add_argument("--repo", default="ReCAP-study")
    parser.add_argument("--branch", default="study-sunday")
    parser.add_argument("--dest-dir", default="outputs/study")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Missing token. Pass --token or set GITHUB_TOKEN.")
    if not args.local_path.exists():
        raise SystemExit(f"File not found: {args.local_path}")

    dest_path = f"{args.dest_dir.rstrip('/')}/{args.local_path.name}"
    url = push_file(
        local_path=args.local_path,
        token=args.token,
        owner=args.owner,
        repo=args.repo,
        branch=args.branch,
        dest_path=dest_path,
        message=f"Add study artifact {args.local_path.name}",
    )
    print(json.dumps({"pushed": dest_path, "url": url}, indent=2))


if __name__ == "__main__":
    main()
