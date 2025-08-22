"""
Tool sandbox for safely executing agent tool calls.
"""

import logging
import json
import time
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import threading
import queue

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by agents."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    timeout: int = 30
    allowed_domains: Optional[List[str]] = None
    max_calls_per_minute: int = 10


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class ToolSandbox:
    """Sandboxed environment for executing agent tool calls."""
    
    def __init__(self, tools: List[Dict[str, Any]] = None):
        """
        Initialize the tool sandbox.
        
        Args:
            tools: List of tool definitions
        """
        self.tools: Dict[str, ToolDefinition] = {}
        self.call_history: List[Dict[str, Any]] = []
        self.rate_limits: Dict[str, List[float]] = {}
        
        # Register default tools
        self._register_default_tools()
        
        # Register custom tools if provided
        if tools:
            for tool_def in tools:
                self.register_tool(tool_def)
    
    def _register_default_tools(self):
        """Register default safe tools."""
        default_tools = [
            {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                },
                "function": self._calculator_tool,
                "timeout": 10
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "function": self._web_search_tool,
                "timeout": 30
            },
            {
                "name": "file_read",
                "description": "Read contents of a file",
                "parameters": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "function": self._file_read_tool,
                "timeout": 10
            },
            {
                "name": "file_write",
                "description": "Write content to a file",
                "parameters": {
                    "path": {"type": "string", "description": "Path to the file",
                            "content": {"type": "string", "description": "Content to write"}}
                },
                "function": self._file_write_tool,
                "timeout": 10
            },
            {
                "name": "system_info",
                "description": "Get system information",
                "parameters": {},
                "function": self._system_info_tool,
                "timeout": 5
            }
        ]
        
        for tool_def in default_tools:
            self.register_tool(tool_def)
    
    def register_tool(self, tool_def: Dict[str, Any]):
        """
        Register a new tool.
        
        Args:
            tool_def: Tool definition dictionary
        """
        try:
            tool = ToolDefinition(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
                function=tool_def["function"],
                timeout=tool_def.get("timeout", 30),
                allowed_domains=tool_def.get("allowed_domains"),
                max_calls_per_minute=tool_def.get("max_calls_per_minute", 10)
            )
            
            self.tools[tool.name] = tool
            self.rate_limits[tool.name] = []
            
            logger.info(f"Registered tool: {tool.name}")
            
        except Exception as e:
            logger.error(f"Error registering tool {tool_def.get('name', 'unknown')}: {e}")
    
    def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls.
        
        Args:
            tool_calls: List of tool call specifications
            
        Returns:
            List of tool results
        """
        results = []
        
        for tool_call in tool_calls:
            try:
                result = self.execute_tool(tool_call)
                results.append({
                    "tool_call_id": tool_call.get("tool_call_id"),
                    "function": tool_call.get("function"),
                    "result": result.result if result.success else None,
                    "error": result.error,
                    "execution_time": result.execution_time,
                    "metadata": result.metadata or {}
                })
                
            except Exception as e:
                logger.error(f"Error executing tool call: {e}")
                results.append({
                    "tool_call_id": tool_call.get("tool_call_id"),
                    "function": tool_call.get("function"),
                    "result": None,
                    "error": str(e),
                    "execution_time": 0.0,
                    "metadata": {}
                })
        
        return results
    
    def execute_tool(self, tool_call: Dict[str, Any]) -> ToolResult:
        """
        Execute a single tool call.
        
        Args:
            tool_call: Tool call specification
            
        Returns:
            Tool execution result
        """
        function_name = tool_call.get("function")
        arguments = tool_call.get("arguments", {})
        
        if function_name not in self.tools:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tool '{function_name}' not found"
            )
        
        tool = self.tools[function_name]
        
        # Check rate limiting
        if not self._check_rate_limit(tool):
            return ToolResult(
                success=False,
                result=None,
                error=f"Rate limit exceeded for tool '{function_name}'"
            )
        
        # Validate arguments
        validation_result = self._validate_arguments(tool, arguments)
        if not validation_result["valid"]:
            return ToolResult(
                success=False,
                result=None,
                error=f"Invalid arguments: {validation_result['error']}"
            )
        
        # Execute tool with timeout
        start_time = time.time()
        
        try:
            # Create a thread for execution with timeout
            result_queue = queue.Queue()
            
            def execute_with_timeout():
                try:
                    result = tool.function(**arguments)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))
            
            thread = threading.Thread(target=execute_with_timeout)
            thread.daemon = True
            thread.start()
            
            # Wait for result with timeout
            thread.join(timeout=tool.timeout)
            
            if thread.is_alive():
                # Tool execution timed out
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Tool execution timed out after {tool.timeout} seconds"
                )
            
            # Get result
            status, result = result_queue.get_nowait()
            
            execution_time = time.time() - start_time
            
            if status == "success":
                # Log successful execution
                self._log_tool_call(tool.name, arguments, execution_time, True)
                
                return ToolResult(
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    metadata={"tool_name": tool.name}
                )
            else:
                # Log failed execution
                self._log_tool_call(tool.name, arguments, execution_time, False, result)
                
                return ToolResult(
                    success=False,
                    result=None,
                    error=result,
                    execution_time=execution_time,
                    metadata={"tool_name": tool.name}
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._log_tool_call(tool.name, arguments, execution_time, False, str(e))
            
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                metadata={"tool_name": tool.name}
            )
    
    def _check_rate_limit(self, tool: ToolDefinition) -> bool:
        """Check if tool call is within rate limits."""
        current_time = time.time()
        call_times = self.rate_limits[tool.name]
        
        # Remove calls older than 1 minute
        call_times = [t for t in call_times if current_time - t < 60]
        self.rate_limits[tool.name] = call_times
        
        # Check if we're under the limit
        if len(call_times) >= tool.max_calls_per_minute:
            return False
        
        # Add current call
        call_times.append(current_time)
        return True
    
    def _validate_arguments(self, tool: ToolDefinition, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool arguments against the tool definition."""
        required_params = tool.parameters
        
        for param_name, param_def in required_params.items():
            if param_name not in arguments:
                return {
                    "valid": False,
                    "error": f"Missing required parameter: {param_name}"
                }
            
            # Type validation (basic)
            expected_type = param_def.get("type", "string")
            actual_value = arguments[param_name]
            
            if expected_type == "string" and not isinstance(actual_value, str):
                return {
                    "valid": False,
                    "error": f"Parameter {param_name} must be a string"
                }
            elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                return {
                    "valid": False,
                    "error": f"Parameter {param_name} must be a number"
                }
        
        return {"valid": True}
    
    def _log_tool_call(self, tool_name: str, arguments: Dict[str, Any], execution_time: float, 
                       success: bool, error: Optional[str] = None):
        """Log a tool call for monitoring and analysis."""
        log_entry = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "arguments": arguments,
            "execution_time": execution_time,
            "success": success,
            "error": error
        }
        
        self.call_history.append(log_entry)
        
        # Keep only last 1000 calls
        if len(self.call_history) > 1000:
            self.call_history = self.call_history[-1000:]
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        tools = []
        
        for tool in self.tools.values():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "timeout": tool.timeout,
                "max_calls_per_minute": tool.max_calls_per_minute
            })
        
        return tools
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools."""
        stats = {}
        
        for tool_name in self.tools.keys():
            tool_calls = [call for call in self.call_history if call["tool_name"] == tool_name]
            
            if tool_calls:
                stats[tool_name] = {
                    "total_calls": len(tool_calls),
                    "successful_calls": sum(1 for call in tool_calls if call["success"]),
                    "failed_calls": sum(1 for call in tool_calls if not call["success"]),
                    "average_execution_time": sum(call["execution_time"] for call in tool_calls) / len(tool_calls),
                    "last_called": max(call["timestamp"] for call in tool_calls) if tool_calls else None
                }
            else:
                stats[tool_name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "average_execution_time": 0.0,
                    "last_called": None
                }
        
        return stats
    
    # Default tool implementations
    
    def _calculator_tool(self, expression: str) -> str:
        """Safe calculator tool."""
        try:
            # Only allow safe mathematical operations
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Expression contains disallowed characters")
            
            # Evaluate the expression
            result = eval(expression)
            return str(result)
            
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")
    
    def _web_search_tool(self, query: str) -> str:
        """Mock web search tool."""
        # In a real implementation, this would use a safe web search API
        return f"Mock search results for: {query}"
    
    def _file_read_tool(self, path: str) -> str:
        """Safe file reading tool."""
        try:
            # Validate path (only allow reading from safe directories)
            safe_dirs = ["/tmp", tempfile.gettempdir()]
            path_obj = Path(path).resolve()
            
            if not any(str(path_obj).startswith(safe_dir) for safe_dir in safe_dirs):
                raise ValueError("Path not in allowed directories")
            
            if not path_obj.exists():
                raise ValueError("File does not exist")
            
            if not path_obj.is_file():
                raise ValueError("Path is not a file")
            
            # Read file content
            with open(path_obj, "r", encoding="utf-8") as f:
                content = f.read()
            
            return content
            
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
    
    def _file_write_tool(self, path: str, content: str) -> str:
        """Safe file writing tool."""
        try:
            # Validate path (only allow writing to safe directories)
            safe_dirs = ["/tmp", tempfile.gettempdir()]
            path_obj = Path(path).resolve()
            
            if not any(str(path_obj).startswith(safe_dir) for safe_dir in safe_dirs):
                raise ValueError("Path not in allowed directories")
            
            # Create directory if it doesn't exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {path}"
            
        except Exception as e:
            raise ValueError(f"Error writing file: {str(e)}")
    
    def _system_info_tool(self) -> str:
        """Get basic system information."""
        import platform
        
        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
        
        return json.dumps(info, indent=2)

