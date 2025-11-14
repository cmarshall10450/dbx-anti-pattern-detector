"""Security event logging for Cloud CEO.

This module provides comprehensive security logging for audit purposes,
including detector loading, prompt injection attempts, and resource violations.
"""

import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class SecurityAuditLogger:
    """Log security-relevant events for monitoring and audit.

    This logger captures security events including:
    - Detector loading with security analysis
    - Prompt injection detection
    - Resource limit violations
    - Timeout events
    """

    def __init__(self) -> None:
        """Initialize security audit logger."""
        self.events: list[dict[str, Any]] = []

    def log_detector_load(
        self,
        detector_path: str,
        analysis: dict[str, Any]
    ) -> None:
        """Log detector loading with security analysis.

        Args:
            detector_path: Path to detector being loaded
            analysis: Security analysis report
        """
        event = {
            'timestamp': time.time(),
            'event_type': 'detector_load',
            'detector': detector_path,
            'risk_level': analysis.get('risk_level', 'unknown'),
            'dangerous_imports': analysis.get('dangerous_imports', []),
            'issue_count': len(analysis.get('issues', []))
        }

        self.events.append(event)

        logger.info(
            "detector_loaded",
            detector=detector_path,
            risk_level=analysis.get('risk_level', 'unknown'),
            dangerous_imports=analysis.get('dangerous_imports', []),
            issue_count=len(analysis.get('issues', []))
        )

        if analysis.get('risk_level') in ['high', 'critical']:
            logger.warning(
                "high_risk_detector_loaded",
                detector=detector_path,
                issues=analysis.get('issues', [])
            )
            self._send_security_alert(event)

    def log_prompt_injection_attempt(
        self,
        message: str,
        patterns: list[str]
    ) -> None:
        """Log potential prompt injection attempts.

        Args:
            message: Message containing injection patterns
            patterns: List of injection patterns detected
        """
        event = {
            'timestamp': time.time(),
            'event_type': 'prompt_injection_attempt',
            'message_preview': message[:100],
            'patterns': patterns
        }

        self.events.append(event)

        logger.warning(
            "prompt_injection_detected",
            patterns_found=patterns,
            message_preview=message[:100]
        )

        self._send_security_alert(event)

    def log_timeout(
        self,
        detector_path: str,
        timeout: int
    ) -> None:
        """Log detector timeout event.

        Args:
            detector_path: Path to detector that timed out
            timeout: Timeout duration in seconds
        """
        event = {
            'timestamp': time.time(),
            'event_type': 'detector_timeout',
            'detector': detector_path,
            'timeout_seconds': timeout
        }

        self.events.append(event)

        logger.error(
            "detector_timeout",
            detector=detector_path,
            timeout_seconds=timeout
        )

    def log_resource_violation(
        self,
        detector: str,
        resource: str,
        limit: float,
        used: float
    ) -> None:
        """Log resource limit violations.

        Args:
            detector: Detector that violated limit
            resource: Resource type (cpu, memory, etc.)
            limit: Limit that was exceeded
            used: Amount used
        """
        event = {
            'timestamp': time.time(),
            'event_type': 'resource_violation',
            'detector': detector,
            'resource': resource,
            'limit': limit,
            'used': used
        }

        self.events.append(event)

        logger.error(
            "resource_violation",
            detector=detector,
            resource=resource,
            limit=limit,
            used=used
        )

    def log_security_validation_passed(
        self,
        detector_path: str,
        analysis: dict[str, Any]
    ) -> None:
        """Log successful security validation.

        Args:
            detector_path: Path to detector that passed validation
            analysis: Security analysis report
        """
        logger.info(
            "security_validation_passed",
            detector=detector_path,
            risk_level=analysis.get('risk_level', 'unknown'),
            complexity=analysis.get('complexity', 0)
        )

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get security events, optionally filtered by type.

        Args:
            event_type: Optional event type filter

        Returns:
            List of security events
        """
        if event_type is None:
            return self.events

        return [e for e in self.events if e['event_type'] == event_type]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about security events.

        Returns:
            Dictionary with event statistics
        """
        stats = {
            'total_events': len(self.events),
            'detector_loads': len(self.get_events('detector_load')),
            'prompt_injections': len(self.get_events('prompt_injection_attempt')),
            'timeouts': len(self.get_events('detector_timeout')),
            'resource_violations': len(self.get_events('resource_violation'))
        }

        high_risk_loads = [
            e for e in self.get_events('detector_load')
            if e.get('risk_level') in ['high', 'critical']
        ]
        stats['high_risk_loads'] = len(high_risk_loads)

        return stats

    def _send_security_alert(self, event: dict[str, Any]) -> None:
        """Send security alert for critical events.

        Args:
            event: Security event to alert on
        """
        logger.critical(
            "SECURITY_ALERT",
            event_type=event['event_type'],
            **{k: v for k, v in event.items() if k not in ['timestamp', 'event_type']}
        )
