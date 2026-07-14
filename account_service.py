"""
account_service.py – Account management: login check, re-login, safe deletion.

Does NOT store passwords, cookies, or tokens.
All authentication is delegated to notebooklm-py CLI / browser.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import database as db
from models import Account, AccountStatus
from settings import DB_PATH

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Login helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_login_commands(profile_name: str) -> list[str]:
    """Return the CLI commands the user should run to authenticate."""
    return [
        f"notebooklm profile create {profile_name}",
        f"# Option A – use existing Edge session (instant, no browser opens):",
        f"notebooklm -p {profile_name} login --browser-cookies edge",
        f"# Option B – open Edge for manual Google sign-in:",
        f"notebooklm -p {profile_name} login --browser msedge",
    ]


def login_with_edge_cookies(profile_name: str) -> tuple[bool, str]:
    """
    Extract auth cookies from an already-signed-in Edge browser session.
    Requires: pip install 'notebooklm-py[cookies]'
    Returns (success, message).
    """
    _ensure_profile(profile_name)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "notebooklm", "-p", profile_name,
             "login", "--browser-cookies", "edge"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0:
            return True, f"کوکی‌های Edge با موفقیت خوانده شد.\n{output}"
        if "notebooklm-py[cookies]" in output or "No module" in output:
            return False, (
                "برای این قابلیت باید پکیج کوکی نصب بشه:\n"
                "pip install \"notebooklm-py[cookies]\""
            )
        return False, output or "خطای ناشناخته"
    except subprocess.TimeoutExpired:
        return False, "عملیات timeout شد."
    except Exception as exc:
        return False, str(exc)


def open_login_browser(profile_name: str) -> tuple[bool, str]:
    """
    Launch notebooklm login with --browser msedge in a visible terminal window.
    Edge is already installed on Windows so no Playwright download is needed.
    Returns (success, message).
    """
    _ensure_profile(profile_name)

    if sys.platform == "win32":
        ps_cmd = f"& '{sys.executable}' -m notebooklm -p {profile_name} login --browser msedge"
        try:
            subprocess.Popen(
                ["powershell", "-NoExit", "-Command", ps_cmd],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            return True, (
                f"پنجره PowerShell برای پروفایل '{profile_name}' باز شد. "
                "در مرورگر Edge که باز می‌شه با گوگل وارد شوید، "
                "سپس به اینجا برگردید و روی 'Check Auth' کلیک کنید."
            )
        except Exception:
            pass

        # Fallback: CMD window
        try:
            subprocess.Popen(
                ["cmd", "/k",
                 f'"{sys.executable}" -m notebooklm -p {profile_name} login --browser msedge'],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            return True, (
                f"پنجره CMD برای پروفایل '{profile_name}' باز شد. "
                "در مرورگر Edge که باز می‌شه با گوگل وارد شوید."
            )
        except Exception as exc:
            return False, f"Failed to open terminal: {exc}"

    else:
        login_args = [sys.executable, "-m", "notebooklm", "-p", profile_name,
                      "login", "--browser", "chrome"]
        for term in (["gnome-terminal", "--"], ["xterm", "-e"], ["konsole", "-e"]):
            try:
                subprocess.Popen(term + login_args)
                return True, f"Terminal opened for '{profile_name}'. Complete Google sign-in then click 'Check Auth'."
            except FileNotFoundError:
                continue
        try:
            subprocess.Popen(login_args)
            return True, "Login process started."
        except FileNotFoundError:
            return False, "notebooklm CLI not found. Run: pip install notebooklm-py"
        except Exception as exc:
            return False, str(exc)


def _ensure_profile(profile_name: str) -> None:
    """Create the notebooklm profile if it doesn't already exist (best-effort)."""
    for candidate in (
        [sys.executable, "-m", "notebooklm"],
        ["notebooklm"],
    ):
        try:
            subprocess.run(
                candidate + ["profile", "create", profile_name],
                timeout=10,
                capture_output=True,
                text=True,
            )
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception:
            return


# ══════════════════════════════════════════════════════════════════════════════
# Auth check
# ══════════════════════════════════════════════════════════════════════════════

async def check_account_auth(
    account: Account,
    path: Path = DB_PATH,
) -> AccountStatus:
    """
    Try to connect to NotebookLM with this profile and return status.
    Updates the account's auth_status in DB.
    """
    db.update_account_auth_status(account.id, AccountStatus.CHECKING, path)

    try:
        from notebooklm import NotebookLMClient  # type: ignore[import]
        async with NotebookLMClient.from_storage(profile=account.profile_name) as client:
            # Minimal operation to verify session
            await client.notebooks.list()
        status = AccountStatus.ACTIVE
    except ImportError:
        logger.warning("notebooklm-py not installed; cannot check auth.")
        status = AccountStatus.ERROR
    except Exception as exc:
        err = str(exc).lower()
        if any(k in err for k in ("auth", "cookie", "401", "403", "login", "expired")):
            status = AccountStatus.AUTH_EXPIRED
        elif any(k in err for k in ("rate", "429", "quota")):
            status = AccountStatus.RATE_LIMITED
        else:
            logger.warning("Auth check failed for %s: %s", account.profile_name, exc)
            status = AccountStatus.ERROR

    db.update_account_auth_status(account.id, status, path)
    return status


# ══════════════════════════════════════════════════════════════════════════════
# Safe deletion checks
# ══════════════════════════════════════════════════════════════════════════════

def get_deletion_risks(account_id: int, path: Path = DB_PATH) -> dict:
    """Return a risk summary before deleting an account."""
    from models import JobStatus
    jobs = []
    try:
        all_projects = db.list_projects(path)
        for p in all_projects:
            for j in db.get_jobs_for_project(p.id, path):
                if j.account_id == account_id:
                    jobs.append(j)
    except Exception:
        pass

    active_jobs = [
        j for j in jobs
        if j.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
    ]
    allocations = []
    try:
        all_projects = db.list_projects(path)
        for p in all_projects:
            for alloc in db.get_allocations_for_project(p.id, path):
                if alloc.account_id == account_id and alloc.enabled:
                    allocations.append(alloc)
    except Exception:
        pass

    return {
        "has_active_jobs": len(active_jobs) > 0,
        "active_job_count": len(active_jobs),
        "has_allocations": len(allocations) > 0,
        "allocation_count": len(allocations),
        "can_delete": len(active_jobs) == 0,
    }


def disable_account(account_id: int, path: Path = DB_PATH) -> None:
    db.update_account(account_id, {"enabled": 0}, path)


def delete_account_record(account_id: int, path: Path = DB_PATH) -> None:
    risks = get_deletion_risks(account_id, path)
    if risks["has_active_jobs"]:
        raise RuntimeError(
            f"Cannot delete account with {risks['active_job_count']} active jobs."
        )
    db.delete_account(account_id, path)


def delete_account_profile_files(profile_name: str) -> tuple[bool, str]:
    """Delete local Chrome/browser profile files for this account."""
    import shutil
    from settings import ACCOUNTS_DIR
    profile_dir = ACCOUNTS_DIR / profile_name
    if profile_dir.exists():
        try:
            shutil.rmtree(profile_dir)
            return True, f"Profile directory '{profile_dir}' deleted."
        except Exception as exc:
            return False, str(exc)
    return True, "No local profile directory found."


# ══════════════════════════════════════════════════════════════════════════════
# Stats reset
# ══════════════════════════════════════════════════════════════════════════════

def reset_account_stats(account_id: int, path: Path = DB_PATH) -> None:
    """Reset allocation counters for this account across all projects."""
    from database import get_db
    with get_db(path) as conn:
        conn.execute(
            """UPDATE project_account_allocations
               SET assigned_jobs_count=0, completed_jobs_count=0, failed_jobs_count=0
               WHERE account_id=?""",
            (account_id,),
        )
