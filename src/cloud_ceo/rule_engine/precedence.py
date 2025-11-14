"""Rule configuration precedence management.

This module implements the precedence logic for resolving rule conflicts
when the same rule_id appears in multiple sources (built-in, YAML, Delta).

Precedence hierarchy (for same rule_id):
    Built-in detector defaults < YAML config < Delta config

Later sources override earlier sources using last-write-wins strategy.
Precedence applies to configuration metadata only (severity, enabled state,
custom thresholds, custom messages). Detection logic lives in detector
classes (Python code) and cannot be overridden via configuration.
"""

from __future__ import annotations

import structlog

from cloud_ceo.rule_engine.schema import RuleSchema

logger = structlog.get_logger(__name__)


class RulePrecedenceManager:
    """Manage rule precedence and conflict resolution.

    Precedence hierarchy (for same rule_id):
        Built-in detector defaults < YAML config < Delta config

    Later sources override earlier sources using last-write-wins strategy.
    Precedence applies to configuration metadata only (severity, enabled state,
    custom thresholds, custom messages). Detection logic lives in detector
    classes (Python code) and cannot be overridden via configuration.

    Example:
        manager = RulePrecedenceManager()

        # Load in precedence order
        manager.add_rules(built_in_rules, "built-in")
        manager.add_rules(yaml_rules, "yaml")
        manager.add_rules(delta_rules, "delta")

        # Get effective configuration
        final_rules = manager.get_all_rules()
    """

    def __init__(self) -> None:
        """Initialize precedence manager with empty rule set."""
        self.rules: dict[str, RuleSchema] = {}
        self.override_audit: dict[str, list[str]] = {}

    def add_rules(self, rules: list[RuleSchema], source_name: str) -> None:
        """Add rules from a source with precedence tracking.

        Args:
            rules: List of RuleSchema objects from this source
            source_name: Source identifier (e.g., "built-in", "yaml", "delta")

        Side effects:
            - Updates self.rules dict with new/overridden rules
            - Tracks override chain in self.override_audit
            - Logs INFO when rules are overridden
        """
        for rule in rules:
            if rule.rule_id in self.rules:
                old_rule = self.rules[rule.rule_id]
                self.override_audit[rule.rule_id].append(source_name)

                changed_fields = []
                if old_rule.severity != rule.severity:
                    changed_fields.append(
                        f"severity: {old_rule.severity.value} → {rule.severity.value}"
                    )
                if old_rule.enabled != rule.enabled:
                    changed_fields.append(
                        f"enabled: {old_rule.enabled} → {rule.enabled}"
                    )
                if old_rule.custom_thresholds != rule.custom_thresholds:
                    changed_fields.append("custom_thresholds: modified")
                if old_rule.custom_message != rule.custom_message:
                    changed_fields.append("custom_message: modified")

                logger.info(
                    "Rule overridden",
                    rule_id=rule.rule_id,
                    source=source_name,
                    changes=", ".join(changed_fields) if changed_fields else "no changes"
                )
            else:
                self.override_audit[rule.rule_id] = [source_name]

            self.rules[rule.rule_id] = rule

    def get_rule(self, rule_id: str) -> RuleSchema | None:
        """Get effective rule configuration by ID.

        Args:
            rule_id: Rule identifier (e.g., "SPARK_001")

        Returns:
            RuleSchema if found, None otherwise
        """
        return self.rules.get(rule_id)

    def get_all_rules(self) -> list[RuleSchema]:
        """Get all active rules after precedence resolution.

        Returns:
            List of RuleSchema objects (deduplicated by rule_id)
        """
        return list(self.rules.values())

    def get_override_history(self, rule_id: str) -> list[str]:
        """Get sources that contributed to this rule.

        Args:
            rule_id: Rule identifier

        Returns:
            List of source names in order loaded (e.g., ["built-in", "yaml", "delta"])
            Empty list if rule not found
        """
        return self.override_audit.get(rule_id, [])

    def get_effective_source(self, rule_id: str) -> str | None:
        """Get the effective source for a rule (last override wins).

        Args:
            rule_id: Rule identifier

        Returns:
            Source name (e.g., "delta") or None if rule not found
        """
        history = self.get_override_history(rule_id)
        return history[-1] if history else None

    def get_stats(self) -> dict[str, int | list[str]]:
        """Get precedence manager statistics for logging/debugging.

        Returns:
            Dict with: total_rules, overridden_rules, sources_used
        """
        overridden_rules = [
            rule_id for rule_id, sources in self.override_audit.items()
            if len(sources) > 1
        ]

        all_sources = set()
        for sources in self.override_audit.values():
            all_sources.update(sources)

        return {
            "total_rules": len(self.rules),
            "overridden_rules": len(overridden_rules),
            "sources_used": sorted(all_sources)
        }
