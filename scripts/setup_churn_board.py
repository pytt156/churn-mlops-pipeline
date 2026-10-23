#!/usr/bin/env python3
"""Create GitHub issues, sub-issues and a three-column project board."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PLAN_FILE = Path(__file__).with_name("churn_board_plan.json")


def command(*args: str, stdin: str | None = None) -> str:
    result = subprocess.run(
        args,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{' '.join(args)} failed: {details}")
    return result.stdout.strip()


def gh_json(*args: str, stdin: str | None = None) -> Any:
    output = command("gh", *args, stdin=stdin)
    return json.loads(output) if output else None


def api(method: str, path: str, data: dict | None = None) -> Any:
    args = ["api", "-X", method, path]
    payload = None
    if data is not None:
        args.extend(["--input", "-"])
        payload = json.dumps(data)
    return gh_json(*args, stdin=payload)


def gql(query: str, variables: dict) -> dict:
    response = gh_json(
        "api",
        "graphql",
        "--input",
        "-",
        stdin=json.dumps({"query": query, "variables": variables}),
    )
    if not isinstance(response, dict):
        raise RuntimeError("GitHub returned an empty GraphQL response")
    if response.get("errors"):
        raise RuntimeError(json.dumps(response["errors"]))
    data = response.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("GitHub returned no GraphQL data")
    return data


def require_dict(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected GitHub response for {label}")
    return value


def get_issues(repo: str) -> dict[str, dict]:
    issues = gh_json(
        "api",
        "--paginate",
        "--slurp",
        f"repos/{repo}/issues?state=all&per_page=100",
    )
    found = {}
    for page in issues or []:
        for issue in page:
            if "pull_request" in issue:
                continue
            for line in (issue.get("body") or "").splitlines():
                if line.startswith("<!-- churn-lab:"):
                    found[line] = issue
                    break
    return found


def tag(key: str) -> str:
    return f"<!-- churn-lab:{key} -->"


def description(key: str, text: str, owner: str, done_when: str | None = None) -> str:
    who = "Whole team" if owner == "team" else f"Person {owner}"
    body = f"{tag(key)}\n\n{text}\n\n**Lead:** {who}"
    if done_when:
        body += f"\n\n**Done when:** {done_when}"
    return body


def issue(
    repo: str, key: str, title: str, body: str, labels: list[str], existing: dict[str, dict]
) -> dict:
    if tag(key) in existing:
        item = existing[tag(key)]
        print(f"Reusing #{item['number']}: {title}")
        return item
    item = require_dict(
        api(
            "POST",
            f"repos/{repo}/issues",
            {
                "title": title,
                "body": body,
                "labels": labels,
            },
        ),
        "new issue",
    )
    existing[tag(key)] = item
    print(f"Created #{item['number']}: {title}")
    return item


def attach_child(repo: str, parent: dict, child: dict) -> None:
    path = f"repos/{repo}/issues/{parent['number']}/sub_issues"
    children = api("GET", f"{path}?per_page=100")
    if any(item["id"] == child["id"] for item in children):
        return
    api("POST", path, {"sub_issue_id": child["id"]})


def get_project(owner: str, title: str) -> dict:
    result = require_dict(
        gh_json(
            "project",
            "list",
            "--owner",
            owner,
            "--format",
            "json",
            "--limit",
            "100",
        ),
        "projects",
    )
    for project in result.get("projects", []):
        if project.get("title") == title:
            print(f"Reusing project: {project['url']}")
            return project
    project = require_dict(
        gh_json(
            "project",
            "create",
            "--owner",
            owner,
            "--title",
            title,
            "--format",
            "json",
        ),
        "created project",
    )
    print(f"Created project: {project['url']}")
    return project


def prepare_project(owner: str, repo: str, project: dict) -> dict:
    number = str(project["number"])
    if not project.get("id"):
        project = require_dict(
            gh_json(
                "project",
                "view",
                number,
                "--owner",
                owner,
                "--format",
                "json",
            ),
            "project view",
        )
    project_id = project.get("id")
    if not isinstance(project_id, str):
        raise RuntimeError("No project GraphQL ID was returned")
    query = """query($id:ID!){node(id:$id){... on ProjectV2{
      fields(first:50){nodes{... on ProjectV2SingleSelectField{
        id name options{id name color description}}}}
      views(first:30){nodes{id name layout}}
    }}}"""
    state = require_dict(gql(query, {"id": project_id}).get("node"), "project")
    status = next(
        (field for field in state["fields"]["nodes"] if field and field.get("name") == "Status"),
        None,
    )
    if status is None:
        raise RuntimeError("Project has no Status field")
    desired = ["Backlog", "In Progress", "Done"]
    current = [option["name"] for option in status["options"]]
    if current != desired:
        items = require_dict(
            gh_json(
                "project",
                "item-list",
                number,
                "--owner",
                owner,
                "--format",
                "json",
                "--limit",
                "100",
            ),
            "project items",
        )
        if items.get("items"):
            raise RuntimeError(
                "Existing project has different Status options and items. "
                "Change its statuses manually before rerunning."
            )
        old = {option["name"]: option for option in status["options"]}
        options = []
        for name, color in zip(desired, ["GRAY", "YELLOW", "GREEN"], strict=False):
            option = {"name": name, "color": color, "description": ""}
            if name in old:
                option["id"] = old[name]["id"]
            options.append(option)
        mutation = """mutation($input:UpdateProjectV2FieldInput!){
          updateProjectV2Field(input:$input){
            projectV2Field{... on ProjectV2SingleSelectField{id}}
          }}"""
        gql(
            mutation,
            {
                "input": {
                    "fieldId": status["id"],
                    "singleSelectOptions": options,
                },
            },
        )
    views = state["views"]["nodes"]
    if not any(v["layout"] == "BOARD_LAYOUT" for v in views):
        mutation = """mutation($input:CreateProjectV2ViewInput!){
          createProjectV2View(input:$input){projectV2View{id}}
        }"""
        gql(
            mutation,
            {
                "input": {
                    "projectId": project_id,
                    "name": "Board",
                    "layout": "BOARD_LAYOUT",
                },
            },
        )
    try:
        command(
            "gh",
            "project",
            "link",
            number,
            "--owner",
            owner,
            "--repo",
            repo,
        )
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            print(f"Project link warning: {exc}", file=sys.stderr)
    return project


def add_to_board(owner: str, project: dict, item: dict) -> None:
    number = str(project["number"])
    result = require_dict(
        gh_json(
            "project",
            "item-list",
            number,
            "--owner",
            owner,
            "--format",
            "json",
            "--limit",
            "200",
        ),
        "project items",
    )
    urls = {row.get("content", {}).get("url") for row in result.get("items", [])}
    if item["html_url"] not in urls:
        command(
            "gh",
            "project",
            "item-add",
            number,
            "--owner",
            owner,
            "--url",
            item["html_url"],
        )
        command(
            "gh",
            "project",
            "item-edit",
            number,
            "--owner",
            owner,
            "--url",
            item["html_url"],
            "--field",
            "Status",
            "--value",
            "Backlog",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    work = plan["work"]
    total = sum(len(item["children"]) for item in work)
    print(f"Repo: {plan['repo']}")
    print(f"Project: {plan['project_title']}")
    print(f"Issues: {len(work)} main, {total} sub-issues")
    for item in work:
        print(f"\n[{item['owner']}] {item['title']}")
        for child in item["children"]:
            print(f"  - {child['title']}")
    if not args.apply:
        print("\nPreview only. Run again with --apply to create.")
        return
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI (gh) is not installed")
    command("gh", "auth", "status")
    gh_json("repo", "view", plan["repo"], "--json", "nameWithOwner")
    project = get_project(plan["owner"], plan["project_title"])
    prepare_project(plan["owner"], plan["repo"], project)
    for name, info in plan["labels"].items():
        command(
            "gh",
            "label",
            "create",
            name,
            "--repo",
            plan["repo"],
            "--color",
            info["color"],
            "--description",
            info["description"],
            "--force",
        )
    existing = get_issues(plan["repo"])
    for group in work:
        owner = group["owner"]
        labels = [group["label"]]
        labels.append("team" if owner == "team" else f"role:{owner}")
        parent = issue(
            plan["repo"],
            group["key"],
            group["title"],
            description(group["key"], group["description"], owner),
            labels,
            existing,
        )
        add_to_board(plan["owner"], project, parent)
        for index, child in enumerate(group["children"], start=1):
            key = f"{group['key']}:{index}"
            item = issue(
                plan["repo"],
                key,
                child["title"],
                description(
                    key,
                    child["description"],
                    owner,
                    child["done_when"],
                ),
                labels,
                existing,
            )
            attach_child(plan["repo"], parent, item)
    print(f"\nDone: {project['url']}")
    print("Only main issues are added to the Kanban board.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
