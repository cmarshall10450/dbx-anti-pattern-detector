"""Static analysis for detector security.

This module performs AST-based security analysis on custom detector code
to detect dangerous imports, function calls, and resource usage patterns.
"""

import ast
from pathlib import Path
from typing import Any

import structlog

from cloud_ceo.rule_engine.exceptions import validate_file_size

logger = structlog.get_logger(__name__)


class DetectorSecurityAnalyzer:
    """Static analysis for detector security.

    Analyzes detector code to identify:
    - Dangerous imports (os, subprocess, socket, etc.)
    - Dangerous function calls (eval, exec, open, etc.)
    - Resource usage patterns (loops, recursion, allocations)
    - Code complexity metrics
    """

    DANGEROUS_IMPORTS = {
        'os', 'sys', 'subprocess', 'socket', 'requests',
        'urllib', 'http', 'ftplib', 'telnetlib', 'ssl',
        'pickle', 'marshal', 'shelve', 'tempfile', 'shutil',
        'importlib', 'imp', 'pkgutil', 'zipimport',
        'ctypes', 'cffi', 'pty', 'rlcompleter'
    }

    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__',
        'open', 'input', 'raw_input', 'execfile',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'globals', 'locals', 'vars', 'dir'
    }

    SUSPICIOUS_PATTERNS = {
        'network_access': ['socket', 'requests', 'urllib', 'http.client', 'ftplib'],
        'file_access': ['open', 'read', 'write', 'Path', 'tempfile'],
        'command_execution': ['subprocess', 'os.system', 'os.popen', 'os.exec'],
        'code_execution': ['eval', 'exec', 'compile', '__import__'],
        'credential_access': ['environ', 'getenv', 'expanduser', 'HOME']
    }

    def __init__(self, max_file_size_mb: int = 10) -> None:
        """Initialize security analyzer.

        Args:
            max_file_size_mb: Maximum allowed file size in MB (default: 10)
        """
        self._max_file_size_mb = max_file_size_mb

    def analyze_detector(self, detector_path: str) -> dict[str, Any]:
        """Perform security analysis on detector code.

        Args:
            detector_path: Path to detector Python file

        Returns:
            Security analysis report with risk level and issues
        """
        try:
            path = Path(detector_path)
            validate_file_size(path, self._max_file_size_mb)
            source = path.read_text()
            return self.analyze_detector_source(source, detector_path)
        except FileNotFoundError:
            logger.error("detector_file_not_found", path=detector_path)
            return {
                'path': detector_path,
                'risk_level': 'unknown',
                'issues': [{'type': 'file_not_found', 'severity': 'critical'}],
                'dangerous_imports': [],
                'suspicious_calls': [],
                'resource_usage': {},
                'complexity': 0
            }
        except Exception as e:
            logger.error("detector_analysis_failed", path=detector_path, error=str(e))
            return {
                'path': detector_path,
                'risk_level': 'unknown',
                'issues': [{'type': 'analysis_error', 'severity': 'high', 'error': str(e)}],
                'dangerous_imports': [],
                'suspicious_calls': [],
                'resource_usage': {},
                'complexity': 0
            }

    def analyze_detector_source(self, source: str, path: str = "unknown") -> dict[str, Any]:
        """Analyze detector source code for security issues.

        Args:
            source: Detector source code
            path: Optional path for reporting

        Returns:
            Security analysis report
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.error("detector_syntax_error", path=path, error=str(e))
            return {
                'path': path,
                'risk_level': 'unknown',
                'issues': [{'type': 'syntax_error', 'severity': 'critical', 'error': str(e)}],
                'dangerous_imports': [],
                'suspicious_calls': [],
                'resource_usage': {},
                'complexity': 0
            }

        report = {
            'path': path,
            'risk_level': 'low',
            'issues': [],
            'dangerous_imports': [],
            'suspicious_calls': [],
            'resource_usage': self._analyze_resource_usage(tree),
            'complexity': self._calculate_complexity(tree)
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in self.DANGEROUS_IMPORTS:
                        report['dangerous_imports'].append(alias.name)
                        # Mark command execution imports as critical
                        severity = 'critical' if module_name in ['os', 'subprocess', 'socket'] else 'high'
                        report['issues'].append({
                            'type': 'dangerous_import',
                            'module': alias.name,
                            'line': node.lineno,
                            'severity': severity
                        })

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in self.DANGEROUS_IMPORTS:
                        report['dangerous_imports'].append(node.module)
                        # Mark command execution imports as critical
                        severity = 'critical' if module_name in ['os', 'subprocess', 'socket'] else 'high'
                        report['issues'].append({
                            'type': 'dangerous_import',
                            'module': node.module,
                            'line': node.lineno,
                            'severity': severity
                        })

            elif isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                if func_name in self.DANGEROUS_FUNCTIONS:
                    report['suspicious_calls'].append(func_name)
                    report['issues'].append({
                        'type': 'dangerous_function',
                        'function': func_name,
                        'line': node.lineno,
                        'severity': 'critical'
                    })

        if any(issue['severity'] == 'critical' for issue in report['issues']):
            report['risk_level'] = 'critical'
        elif any(issue['severity'] == 'high' for issue in report['issues']):
            report['risk_level'] = 'high'
        elif len(report['issues']) > 3:
            report['risk_level'] = 'medium'

        if report['resource_usage'].get('risk') == 'high':
            if report['risk_level'] == 'low':
                report['risk_level'] = 'medium'

        return report

    def _analyze_resource_usage(self, tree: ast.AST) -> dict[str, Any]:
        """Analyze potential resource usage patterns.

        Args:
            tree: AST tree to analyze

        Returns:
            Resource usage analysis
        """
        loops = 0
        recursion = False
        allocations = 0
        nested_loops = 0

        class LoopVisitor(ast.NodeVisitor):
            def __init__(self):
                self.loop_depth = 0
                self.max_depth = 0

            def visit_For(self, node):
                self.loop_depth += 1
                self.max_depth = max(self.max_depth, self.loop_depth)
                self.generic_visit(node)
                self.loop_depth -= 1

            def visit_While(self, node):
                self.loop_depth += 1
                self.max_depth = max(self.max_depth, self.loop_depth)
                self.generic_visit(node)
                self.loop_depth -= 1

        visitor = LoopVisitor()
        visitor.visit(tree)
        nested_loops = visitor.max_depth

        for node in ast.walk(tree):
            if isinstance(node, (ast.While, ast.For)):
                loops += 1
            elif isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func_name = self._get_function_name(child)
                        if func_name == node.name:
                            recursion = True
            elif isinstance(node, (ast.List, ast.Dict, ast.Set)):
                allocations += 1

        risk = 'low'
        if nested_loops > 2 or (loops > 5 and recursion):
            risk = 'high'
        elif loops > 3 or recursion:
            risk = 'medium'

        return {
            'loops': loops,
            'has_recursion': recursion,
            'allocations': allocations,
            'nested_loops': nested_loops,
            'risk': risk
        }

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity.

        Args:
            tree: AST tree to analyze

        Returns:
            Cyclomatic complexity score
        """
        complexity = 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1

        return complexity

    def _get_function_name(self, node: ast.Call) -> str:
        """Extract function name from call node.

        Args:
            node: AST Call node

        Returns:
            Function name or 'unknown'
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return 'unknown'

    def generate_report(self, detector_path: str) -> str:
        """Generate human-readable security report.

        Args:
            detector_path: Path to detector file

        Returns:
            Formatted security report
        """
        analysis = self.analyze_detector(detector_path)

        report = f"""
# Security Analysis Report for {analysis['path']}

## Risk Level: {analysis['risk_level'].upper()}

## Issues Found: {len(analysis['issues'])}

### Dangerous Imports:
{', '.join(analysis['dangerous_imports']) if analysis['dangerous_imports'] else 'None'}

### Suspicious Function Calls:
{', '.join(analysis['suspicious_calls']) if analysis['suspicious_calls'] else 'None'}

### Resource Usage Analysis:
- Loops detected: {analysis['resource_usage'].get('loops', 0)}
- Nested loops: {analysis['resource_usage'].get('nested_loops', 0)}
- Recursion detected: {analysis['resource_usage'].get('has_recursion', False)}
- Memory allocations: {analysis['resource_usage'].get('allocations', 0)}
- Resource risk: {analysis['resource_usage'].get('risk', 'unknown')}

### Code Complexity:
- Cyclomatic complexity: {analysis['complexity']}

## Detailed Issues:
"""
        for issue in analysis['issues']:
            report += f"""
- **{issue['severity'].upper()}**: {issue['type']}
  - Line {issue.get('line', 'unknown')}: {issue.get('module') or issue.get('function', 'unknown')}
"""

        if not analysis['issues']:
            report += "\nNo issues detected.\n"

        return report
