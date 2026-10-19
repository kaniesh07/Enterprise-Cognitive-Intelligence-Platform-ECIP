"""
tools/approval_gate.py
-----------------------
Human-in-the-loop approval gate for HIGH_RISK actions.

Flow:
  1. Agent proposes action  →  approval_gate.request()  →  stored as PENDING
  2. Human reviews via GET /approvals
  3. Human approves/rejects via POST /approvals/{id}/decision
  4. Agent polls or webhook triggers execution
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import structlog

logger = structlog.get_logger()

class ApprovalStatus(str, Enum):
    PENDING  = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED  = "expired"

@dataclass
class ApprovalRequest:
    request_id:   str
    action_type:  str
    params:       dict
    proposed_by:  str               # agent name that proposed this
    risk_reason:  str
    status:       ApprovalStatus = ApprovalStatus.PENDING
    created_at:   float = field(default_factory=time.time)
    decided_at:   Optional[float] = None
    decided_by:   Optional[str]  = None
    expires_in_s: float = 3600.0   # auto-expire after 1 hour

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.expires_in_s

    def to_dict(self) -> dict:
        return {
            "request_id":  self.request_id,
            "action_type": self.action_type,
            "params":      self.params,
            "proposed_by": self.proposed_by,
            "risk_reason": self.risk_reason,
            "status":      self.status.value,
            "created_at":  self.created_at,
            "decided_at":  self.decided_at,
            "decided_by":  self.decided_by,
        }


class ApprovalGate:
    """
    In-memory approval queue.
    Production: swap dict for PostgreSQL table or Redis sorted set.
    """
    def __init__(self):
        self._requests: dict[str, ApprovalRequest] = {}

    def request(
        self,
        action_type: str,
        params: dict,
        proposed_by: str,
        risk_reason: str,
    ) -> ApprovalRequest:
        req = ApprovalRequest(
            request_id  = str(uuid.uuid4()),
            action_type = action_type,
            params      = params,
            proposed_by = proposed_by,
            risk_reason = risk_reason,
        )
        self._requests[req.request_id] = req
        logger.warning(
            "approval_requested",
            request_id=req.request_id,
            action=action_type,
            proposed_by=proposed_by,
        )
        return req

    def decide(self, request_id: str, approved: bool, decided_by: str) -> ApprovalRequest:
        req = self._requests.get(request_id)
        if not req:
            raise ValueError(f"Approval request {request_id} not found.")
        if req.is_expired:
            req.status = ApprovalStatus.EXPIRED
            raise ValueError(f"Approval request {request_id} has expired.")
        req.status     = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        req.decided_at = time.time()
        req.decided_by = decided_by
        logger.info("approval_decided", request_id=request_id,
                    approved=approved, decided_by=decided_by)
        return req

    def get(self, request_id: str) -> Optional[ApprovalRequest]:
        req = self._requests.get(request_id)
        if req and req.is_expired and req.status == ApprovalStatus.PENDING:
            req.status = ApprovalStatus.EXPIRED
        return req

    def list_pending(self) -> list[ApprovalRequest]:
        for req in self._requests.values():
            if req.is_expired and req.status == ApprovalStatus.PENDING:
                req.status = ApprovalStatus.EXPIRED
        return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

    def stats(self) -> dict:
        statuses = [r.status.value for r in self._requests.values()]
        return {s: statuses.count(s) for s in set(statuses)}


_gate: Optional[ApprovalGate] = None

def get_approval_gate() -> ApprovalGate:
    global _gate
    if _gate is None:
        _gate = ApprovalGate()
    return _gate
