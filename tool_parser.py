"""
Tool calling parser for Qwen3 multi-GPU server
Converts Qwen3's XML tool calls to OpenAI JSON format
"""

import json
import logging
import re
import uuid
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger("qwen3_server.tool_parser")


class Qwen3ToolParser:
    """Parser for Qwen3 XML tool calls"""
    
    def __init__(self):
        # Regex patterns for XML parsing
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )
        # Remove complex fallback patterns - Jinja template should enforce canonical format
        
        # Track parsing statistics
        self.stats = {
            "total_calls": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "malformed_xml": 0
        }
    
    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from Qwen3 response text.
        With Jinja template enforcement, we expect only canonical
        <tool_call><function=name><parameter=name>value</parameter></function></tool_call> format.
        """
        if not text:
            return []

        tool_calls: List[Dict[str, Any]] = []

        # Find canonical <tool_call> wrapper blocks
        tool_call_matches = self.tool_call_regex.findall(text)
        for index, match in enumerate(tool_call_matches):
            tool_call_content = match[0] if match[0] else match[1]
            if not tool_call_content.strip():
                continue
            try:
                tc = self._parse_xml_function_call(tool_call_content, index)
                if tc:
                    tool_calls.append(tc)
                    self.stats["successful_parses"] += 1
                else:
                    self.stats["failed_parses"] += 1
            except Exception as e:
                logger.warning(f"Failed to parse tool call: {e}")
                self.stats["malformed_xml"] += 1

        self.stats["total_calls"] += len(tool_calls)
        return tool_calls
    
    def _parse_xml_function_call(self, xml_content: str, index: int = 0) -> Optional[Dict[str, Any]]:
        """
        Parse a single XML function call
        
        Args:
            xml_content: XML content for a single function call
            
        Returns:
            OpenAI-compatible tool call dictionary or None
        """
        try:
            # Extract function name from <function=name> tag
            function_match = self.tool_call_function_regex.search(xml_content)
            if not function_match:
                logger.warning("No function name found in XML")
                return None
            
            function_content = function_match.group(1) if function_match.group(1) else function_match.group(2)
            if not function_content:
                logger.warning("Empty function content in XML")
                return None
            
            # Function name is the first line/word in the function content
            function_name = function_content.split('\n')[0].strip().split('>')[0]
            if not function_name:
                logger.warning("Could not extract function name")
                return None
                
            # Validate function name is reasonable (alphanumeric + underscore)
            if not function_name.replace('_', '').replace('-', '').isalnum():
                logger.warning(f"Invalid function name: {function_name}")
                return None
            
            # Extract parameters
            parameters = {}
            param_matches = self.tool_call_parameter_regex.findall(xml_content)
            
            for param_match in param_matches:
                param_content = param_match[0] if param_match[0] else param_match[1]
                
                # Parse parameter name and value from <parameter=name>value</parameter>
                param_parts = param_content.split('\n', 1)
                if len(param_parts) >= 2:
                    param_name = param_parts[0].strip().split('>')[0]  # Remove any trailing >
                    param_value = param_parts[1].strip()
                    
                    # Validate parameter name
                    if param_name and param_name.replace('_', '').replace('-', '').isalnum():
                        # Convert parameter value to appropriate type
                        converted_value = self._convert_param_value(param_value)
                        parameters[param_name] = converted_value
                    else:
                        logger.warning(f"Invalid parameter name: {param_name}")
                else:
                    logger.warning(f"Malformed parameter content: {param_content[:50]}...")
            
            # Create OpenAI-compatible tool call
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "index": index,
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(parameters, ensure_ascii=False)
                }
            }
            
            return tool_call
            
        except Exception as e:
            logger.error(f"Error parsing XML function call: {e}")
            return None
    
    # Removed obsolete parsing methods - Jinja template enforces canonical format

    def _convert_param_value(self, value: str) -> Union[str, int, float, bool, None]:
        """
        Convert parameter value to appropriate type
        
        Args:
            value: String parameter value
            
        Returns:
            Converted value with appropriate type
        """
        if not value:
            return None
        
        value = value.strip()
        
        # Try to convert to boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try to convert to integer
        try:
            if '.' not in value and value.replace('-', '').isdigit():
                return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string (default)
        return value
    
    def clean_text(self, text: str) -> str:
        """
        Remove tool calls from text to get clean response
        
        Args:
            text: Text containing tool calls
            
        Returns:
            Clean text without tool calls
        """
        if not text:
            return text
        
        # Remove canonical <tool_call> blocks only
        cleaned_text = self.tool_call_regex.sub('', text)

        # Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text
    
    def get_parsing_stats(self) -> Dict[str, int]:
        """
        Get parsing statistics
        
        Returns:
            Dictionary with parsing statistics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset parsing statistics"""
        self.stats = {
            "total_calls": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "malformed_xml": 0
        }


class ToolCallValidator:
    """Validator for tool calls"""
    
    def __init__(self):
        self.logger = logging.getLogger("qwen3_server.tool_validator")
    
    def validate_tool_call(self, tool_call: Dict[str, Any], tool_schemas: List[Dict[str, Any]]) -> bool:
        """
        Validate a tool call against available tool schemas
        
        Args:
            tool_call: Tool call dictionary
            tool_schemas: List of available tool schemas
            
        Returns:
            True if valid, False otherwise
        """
        try:
            function_name = tool_call.get("function", {}).get("name")
            if not function_name:
                self.logger.warning("Tool call missing function name")
                return False
            
            # Find matching schema
            schema = None
            for tool_schema in tool_schemas:
                if tool_schema.get("function", {}).get("name") == function_name:
                    schema = tool_schema
                    break
            
            if not schema:
                self.logger.warning(f"Unknown function: {function_name}")
                return False
            
            # Validate parameters
            arguments = tool_call.get("function", {}).get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON arguments for {function_name}")
                    return False
            
            return self._validate_parameters(arguments, schema.get("function", {}).get("parameters", {}))
            
        except Exception as e:
            self.logger.error(f"Error validating tool call: {e}")
            return False
    
    def _validate_parameters(self, arguments: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate parameters against schema
        
        Args:
            arguments: Function arguments
            schema: Parameter schema
            
        Returns:
            True if valid, False otherwise
        """
        try:
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Check required parameters
            for param_name in required:
                if param_name not in arguments:
                    self.logger.warning(f"Missing required parameter: {param_name}")
                    return False
            
            # Check parameter types
            for param_name, param_value in arguments.items():
                if param_name not in properties:
                    self.logger.warning(f"Unknown parameter: {param_name}")
                    return False
                
                param_schema = properties[param_name]
                param_type = param_schema.get("type")
                
                if not self._validate_parameter_type(param_value, param_type):
                    self.logger.warning(f"Invalid type for parameter {param_name}: expected {param_type}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating parameters: {e}")
            return False
    
    def _validate_parameter_type(self, value: Any, expected_type: str) -> bool:
        """
        Validate parameter type
        
        Args:
            value: Parameter value
            expected_type: Expected type string
            
        Returns:
            True if type matches, False otherwise
        """
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        else:
            # Unknown type, accept any value
            return True


# Global instances
tool_parser = Qwen3ToolParser()
tool_validator = ToolCallValidator()


def get_tool_parser() -> Qwen3ToolParser:
    """Get the global tool parser instance"""
    return tool_parser


def get_tool_validator() -> ToolCallValidator:
    """Get the global tool validator instance"""
    return tool_validator 