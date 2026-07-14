"""
allocation_service.py – Job distribution and quota management.

Handles both EXACT and FLEXIBLE allocation modes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import database as db
from models import (
    AccountStatus, AllocationMode,
)
from settings import DB_PATH

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_allocations(project_id: int, path: Path = DB_PATH) -> dict:
    """
    Check if total quota covers all chunks.
    Returns a report dict with surplus/deficit information.
    """
    chunks = db.get_chunks_for_project(project_id, path)
    total_chunks = len(chunks)

    allocations = db.get_allocations_for_project(project_id, path)
    enabled = [a for a in allocations if a.enabled]

    total_quota = sum(a.max_jobs_for_project for a in enabled)
    deficit = max(0, total_chunks - total_quota)
    surplus = max(0, total_quota - total_chunks)

    issues: list[str] = []
    warnings: list[str] = []

    if deficit > 0:
        issues.append(f"{deficit} chunk(s) will have no assigned account.")
    if surplus > 0:
        warnings.append(f"{surplus} extra quota slot(s) available.")
    if total_chunks == 0:
        issues.append("No chunks defined.")
    if not enabled:
        issues.append("No accounts enabled for this project.")

    for alloc in enabled:
        account = db.get_account(alloc.account_id, path)
        if account and account.auth_status.value not in ("ACTIVE",):
            warnings.append(
                f"Account '{account.display_name}' connection status: {account.auth_status.value}"
            )

    return {
        "total_chunks": total_chunks,
        "total_quota": total_quota,
        "deficit": deficit,
        "surplus": surplus,
        "enabled_accounts": len(enabled),
        "can_start": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
    }


def validate_project_preflight(
    project_id: int,
    path: Path = DB_PATH,
    max_sources_per_notebook: int = 50,
) -> dict:
    """Validate everything required before starting a NotebookLM project.

    This is UI-independent so callers can present the structured issues and
    warnings wherever they need them.
    """
    project = db.get_project(project_id, path)
    chunks = db.get_chunks_for_project(project_id, path)
    allocations = db.get_allocations_for_project(project_id, path)
    accounts = {a.id: a for a in db.list_accounts(path)}

    active_allocations = [
        alloc for alloc in allocations
        if alloc.enabled
        and (account := accounts.get(alloc.account_id)) is not None
        and account.enabled
        and account.auth_status == AccountStatus.ACTIVE
    ]
    active_allocations.sort(key=lambda alloc: (-alloc.priority, alloc.id))

    issues: list[str] = []
    warnings: list[str] = []
    if project is None:
        issues.append(f"Project {project_id} was not found.")
    if not chunks:
        issues.append("No chunks defined.")
    if not active_allocations:
        issues.append("No active account allocations are available.")

    inactive_enabled = [
        alloc for alloc in allocations if alloc.enabled and alloc not in active_allocations
    ]
    if inactive_enabled:
        warnings.append(
            f"{len(inactive_enabled)} enabled allocation(s) use inactive or disabled accounts."
        )

    missing_chunk_ids = [
        chunk.id for chunk in chunks
        if not chunk.pdf_path or not Path(chunk.pdf_path).is_file()
    ]
    if missing_chunk_ids:
        issues.append(f"{len(missing_chunk_ids)} chunk PDF file(s) are missing.")

    total_quota = sum(alloc.max_jobs_for_project for alloc in active_allocations)
    allocation_deficit = max(0, len(chunks) - total_quota)
    if allocation_deficit and (
        project is None or project.allocation_mode == AllocationMode.EXACT
    ):
        issues.append(
            f"Active allocation quota is short by {allocation_deficit} job(s)."
        )
    elif allocation_deficit:
        warnings.append(
            f"Active allocation quota is short by {allocation_deficit} job(s); "
            "FLEXIBLE overflow will be required."
        )

    proposed_accounts: dict[int, int] = {}
    remaining = {
        alloc.account_id: alloc.max_jobs_for_project for alloc in active_allocations
    }
    overflow_index = 0
    for chunk in chunks:
        target = next(
            (
                alloc for alloc in active_allocations
                if remaining[alloc.account_id] > 0
            ),
            None,
        )
        if target is not None:
            remaining[target.account_id] -= 1
            proposed_accounts[chunk.id] = target.account_id
        elif (
            project is not None
            and project.allocation_mode == AllocationMode.FLEXIBLE
            and active_allocations
        ):
            target = active_allocations[overflow_index % len(active_allocations)]
            overflow_index += 1
            proposed_accounts[chunk.id] = target.account_id

    source_counts: dict[int, int] = {}
    missing_shared_source_ids: set[int] = set()
    if project is not None:
        from shared_source_service import get_sources_for_notebook

        for chunk in chunks:
            account_id = (
                chunk.assigned_account_id
                or proposed_accounts.get(chunk.id)
                or 0
            )
            shared_sources = get_sources_for_notebook(
                project_id, chunk.id, account_id, path
            )
            source_counts[chunk.id] = 1 + len(shared_sources)
            missing_shared_source_ids.update(
                source.id
                for source in shared_sources
                if not source.file_path or not Path(source.file_path).is_file()
            )

    if missing_shared_source_ids:
        issues.append(
            f"{len(missing_shared_source_ids)} shared source file(s) are missing."
        )

    over_source_limit = {
        chunk_id: count for chunk_id, count in source_counts.items()
        if count > max_sources_per_notebook
    }
    if over_source_limit:
        issues.append(
            f"{len(over_source_limit)} notebook(s) exceed the "
            f"{max_sources_per_notebook}-source limit."
        )

    return {
        "project_id": project_id,
        "chunk_count": len(chunks),
        "active_account_count": len(
            {alloc.account_id for alloc in active_allocations}
        ),
        "active_allocation_count": len(active_allocations),
        "total_active_quota": total_quota,
        "allocation_deficit": allocation_deficit,
        "missing_chunk_ids": missing_chunk_ids,
        "source_counts": source_counts,
        "missing_shared_source_ids": sorted(missing_shared_source_ids),
        "over_source_limit": over_source_limit,
        "can_start": not issues,
        "issues": issues,
        "warnings": warnings,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Distribution
# ══════════════════════════════════════════════════════════════════════════════

def distribute_chunks(
    project_id: int,
    mode: AllocationMode = AllocationMode.EXACT,
    path: Path = DB_PATH,
) -> dict[int, list[int]]:
    """
    Assign chunk IDs to account IDs according to quota.

    Returns: {account_id: [chunk_id, ...]}
    """
    chunks = db.get_chunks_for_project(project_id, path)
    if not chunks:
        return {}

    allocations = db.get_allocations_for_project(project_id, path)
    enabled = [a for a in allocations if a.enabled]
    if not enabled:
        raise RuntimeError("No accounts enabled for this project.")

    # Sort by priority desc, then id
    enabled.sort(key=lambda a: (-a.priority, a.id))

    result: dict[int, list[int]] = {a.account_id: [] for a in enabled}
    quota = {a.account_id: a.max_jobs_for_project for a in enabled}
    unassigned: list[int] = []

    for chunk in chunks:
        assigned = False
        for alloc in enabled:
            if quota[alloc.account_id] > 0:
                result[alloc.account_id].append(chunk.id)
                quota[alloc.account_id] -= 1
                assigned = True
                break
        if not assigned:
            unassigned.append(chunk.id)

    if unassigned:
        if mode == AllocationMode.FLEXIBLE:
            # Distribute overflow round-robin
            cycle_accounts = enabled
            for i, chunk_id in enumerate(unassigned):
                acc = cycle_accounts[i % len(cycle_accounts)]
                result[acc.account_id].append(chunk_id)
        else:
            logger.warning(
                "EXACT mode: %d chunks unassigned (insufficient quota).",
                len(unassigned),
            )

    return result


def apply_distribution(
    project_id: int,
    distribution: dict[int, list[int]],
    path: Path = DB_PATH,
) -> None:
    """
    Write assigned_account_id to chunks and create jobs.
    Idempotent: won't create duplicate jobs for already-assigned chunks.
    """
    existing_jobs = {j.chunk_id: j for j in db.get_jobs_for_project(project_id, path)}

    for account_id, chunk_ids in distribution.items():
        for chunk_id in chunk_ids:
            db.update_chunk(chunk_id, {"assigned_account_id": account_id}, path)

            if chunk_id not in existing_jobs:
                job_id, created = db.create_job_if_absent(
                    project_id, chunk_id, account_id, path
                )
                if created:
                    db.increment_allocation_counter(
                        project_id, account_id, "assigned_jobs_count", path
                    )
                existing_jobs[chunk_id] = db.get_job(job_id, path)


def get_jobs_per_account(
    project_id: int, path: Path = DB_PATH
) -> dict[int, list]:
    """Return {account_id: [job, ...]} for all pending/active jobs."""
    jobs = db.get_pending_jobs(project_id, path)
    result: dict[int, list] = {}
    for job in jobs:
        acc_id = job.account_id or 0
        result.setdefault(acc_id, []).append(job)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Manual override
# ══════════════════════════════════════════════════════════════════════════════

def reassign_job(
    job_id: int,
    new_account_id: int,
    path: Path = DB_PATH,
) -> None:
    """Manually reassign a PENDING job to a different account."""
    from models import JobStatus
    job = db.get_job(job_id, path)
    if not job:
        raise ValueError(f"Job {job_id} not found.")
    if job.status not in (JobStatus.PENDING, JobStatus.FAILED):
        raise RuntimeError(f"Can only reassign PENDING/FAILED jobs, not {job.status.value}.")
    db.reset_job_for_retry(job_id, new_account_id, path)
    db.update_chunk(job.chunk_id, {"assigned_account_id": new_account_id}, path)


def reassign_from_failed_account(
    project_id: int,
    failed_account_id: int,
    mode: AllocationMode = AllocationMode.EXACT,
    path: Path = DB_PATH,
) -> int:
    """
    In FLEXIBLE mode, move unstarted jobs from a failed account to healthy ones.
    Returns number of jobs reassigned.
    """
    if mode == AllocationMode.EXACT:
        logger.info("EXACT mode: jobs waiting for re-login on account %d.", failed_account_id)
        return 0

    from models import JobStatus
    jobs = db.get_jobs_for_project(project_id, path)
    to_reassign = [
        j for j in jobs
        if j.account_id == failed_account_id
        and j.status in (JobStatus.PENDING, JobStatus.ASSIGNED)
    ]
    if not to_reassign:
        return 0

    healthy = [
        a for a in db.get_active_accounts(path)
        if a.id != failed_account_id
    ]
    if not healthy:
        logger.error("No healthy accounts to reassign to.")
        return 0

    for i, job in enumerate(to_reassign):
        target = healthy[i % len(healthy)]
        db.reset_job_for_retry(job.id, target.id, path)
        db.update_chunk(job.chunk_id, {"assigned_account_id": target.id}, path)

    logger.info("Reassigned %d jobs from account %d.", len(to_reassign), failed_account_id)
    return len(to_reassign)
