import asyncio
import json
import uuid
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List
from pathlib import Path

from agent.agentSession import AgentSession
from agent.model_manager import ModelManager
from utils.utils import log_step, log_error, log_json_block
from utils.json_parser import parse_llm_json
from mcp_servers.multiMCP import MultiMCP


class BrowserAgent:
    """
    Standalone BrowserAgent for multi-step browser automation tasks.
    Takes a single instruction and operates autonomously across multiple steps.
    Completely decoupled from the Decision agent.
    """
    
    def __init__(self, browser_prompt_path: str = "prompts/browser_agent_prompt.txt", multi_mcp: Optional[MultiMCP] = None):
        self.browser_prompt_path = browser_prompt_path
        self.multi_mcp = multi_mcp
        self.model = ModelManager()
        self.max_steps = 15
        self.max_retries = 3
        
    async def run(self, instruction: str, session: Optional[AgentSession] = None) -> Dict[str, Any]:
        """
        Main entry point for BrowserAgent.
        Takes an instruction and executes multi-step browser automation.
        """
        session_id = str(uuid.uuid4())
        log_step(f"🚀 BrowserAgent starting with instruction: {instruction[:100]}...", symbol="🌐")
        
        # Initialize session if not provided
        if not session:
            session = AgentSession(session_id, instruction)
        
        # Initialize execution state
        execution_state = {
            "instruction": instruction,
            "current_step": 0,
            "completed_steps": [],
            "failed_steps": [],
            "browser_state": {},
            "final_result": None,
            "status": "in_progress"
        }
        
        try:
            # Main execution loop
            while execution_state["current_step"] < self.max_steps and execution_state["status"] == "in_progress":
                step_result = await self._execute_step(execution_state, session)
                
                if step_result["status"] == "success":
                    execution_state["completed_steps"].append(step_result)
                    execution_state["current_step"] += 1
                    
                    # Check if task is complete
                    if step_result.get("task_complete", False):
                        execution_state["status"] = "completed"
                        execution_state["final_result"] = step_result["result"]
                        break
                        
                elif step_result["status"] == "error":
                    execution_state["failed_steps"].append(step_result)
                    
                    # Retry logic
                    if len(execution_state["failed_steps"]) >= self.max_retries:
                        execution_state["status"] = "failed"
                        break
                    
                    # Continue to next step despite error
                    execution_state["current_step"] += 1
                    
            # Final status check
            if execution_state["status"] == "in_progress":
                execution_state["status"] = "timeout"
                
            log_step(f"✅ BrowserAgent completed with status: {execution_state['status']}", symbol="🌐")
            
            # Ensure we have a proper result message
            if execution_state["final_result"] is None:
                if execution_state["status"] == "failed":
                    execution_state["final_result"] = "BrowserAgent failed due to repeated errors"
                elif execution_state["status"] == "timeout":
                    execution_state["final_result"] = "BrowserAgent timed out after maximum steps"
                else:
                    execution_state["final_result"] = "BrowserAgent completed without final result"
            
            return {
                "status": execution_state["status"],
                "result": execution_state["final_result"],
                "steps_completed": len(execution_state["completed_steps"]),
                "steps_failed": len(execution_state["failed_steps"]),
                "total_steps": execution_state["current_step"]
            }
            
        except Exception as e:
            log_error("BrowserAgent execution failed", e)
            return {
                "status": "error",
                "result": f"BrowserAgent failed: {str(e)}",
                "steps_completed": len(execution_state["completed_steps"]),
                "steps_failed": len(execution_state["failed_steps"]),
                "total_steps": execution_state["current_step"]
            }
    
    async def _execute_step(self, execution_state: Dict[str, Any], session: AgentSession) -> Dict[str, Any]:
        """
        Execute a single step in the browser automation process.
        """
        step_number = execution_state["current_step"]
        log_step(f"🔧 Executing BrowserAgent step {step_number + 1}", symbol="🌐")
        
        # Build step input
        step_input = self._build_step_input(execution_state, step_number)
        
        # Get step plan from LLM
        step_plan = await self._get_step_plan(step_input, session)
        
        if not step_plan:
            return {
                "status": "error",
                "result": "Failed to generate step plan",
                "step_number": step_number
            }
        
        # Execute the step using MCP tools
        try:
            step_result = await self._execute_step_plan(step_plan, execution_state, session)
            return step_result
            
        except Exception as e:
            log_error(f"Step {step_number + 1} execution failed", e)
            return {
                "status": "error",
                "result": f"Step execution failed: {str(e)}",
                "step_number": step_number
            }
    
    def _build_step_input(self, execution_state: Dict[str, Any], step_number: int) -> Dict[str, Any]:
        """
        Build input for the current step.
        """
        return {
            "current_time": datetime.now(UTC).isoformat(),
            "step_number": step_number + 1,
            "max_steps": self.max_steps,
            "original_instruction": execution_state["instruction"],
            "completed_steps": execution_state["completed_steps"],
            "failed_steps": execution_state["failed_steps"],
            "browser_state": execution_state["browser_state"],
            "available_tools": self._get_available_tools()
        }
    
    async def _get_step_plan(self, step_input: Dict[str, Any], session: AgentSession) -> Optional[Dict[str, Any]]:
        """
        Get step plan from LLM using browser agent prompt.
        """
        try:
            prompt_template = Path(self.browser_prompt_path).read_text(encoding="utf-8")
            full_prompt = (
                f"{prompt_template.strip()}\n\n"
                "```json\n"
                f"{json.dumps(step_input, indent=2)}\n"
                "```"
            )
            
            log_step("[SENDING PROMPT TO BROWSER AGENT...]", symbol="🌐")
            
            response = await self.model.generate_text(prompt=full_prompt)
            
            log_step("[RECEIVED OUTPUT FROM BROWSER AGENT...]", symbol="🌐")
            
            # Parse the response
            step_plan = parse_llm_json(response, required_keys=[
                'action_type', 'tool_name', 'parameters', 'reasoning', 'expected_outcome'
            ])
            
            log_json_block(f"📌 BrowserAgent Step Plan", step_plan)
            
            return step_plan
            
        except Exception as e:
            log_error("Failed to get step plan from LLM", e)
            return None
    
    async def _execute_step_plan(self, step_plan: Dict[str, Any], execution_state: Dict[str, Any], session: AgentSession) -> Dict[str, Any]:
        """
        Execute the step plan using MCP tools.
        """
        if not self.multi_mcp:
            raise RuntimeError("MultiMCP not available for tool execution")
        
        tool_name = step_plan["tool_name"]
        parameters = step_plan["parameters"]
        
        log_step(f"🔧 Executing tool: {tool_name} with params: {parameters}", symbol="🌐")
        
        try:
            # Execute the tool with retry logic for browser context errors
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Call the MCP tool with proper argument format
                    # Convert parameters to positional arguments for function_wrapper
                    if parameters:
                        # Get the parameter values in the order they appear in the schema
                        tool_entry = self.multi_mcp.tool_map.get(tool_name)
                        if tool_entry:
                            tool = tool_entry["tool"]
                            schema = tool.inputSchema
                            
                            if "input" in schema.get("properties", {}):
                                # Nested schema case
                                inner_key = next(iter(schema.get("$defs", {})), None)
                                if inner_key:
                                    inner_props = schema["$defs"][inner_key]["properties"]
                                    param_names = list(inner_props.keys())
                                    args = [parameters.get(name) for name in param_names]
                                else:
                                    args = list(parameters.values())
                            else:
                                # Direct schema case
                                param_names = list(schema["properties"].keys())
                                args = [parameters.get(name) for name in param_names]
                        else:
                            # Fallback: use parameter values in order
                            args = list(parameters.values())
                    else:
                        args = []
                    
                    result = await self.multi_mcp.function_wrapper(tool_name, *args)
                    
                    # Serialize the result for storage
                    serialized_result = self._serialize_result(result)
                    
                    # Store the result
                    execution_state["browser_state"].update({
                        "last_action": tool_name,
                        "last_result": serialized_result,
                        "current_url": self._extract_url_from_result(serialized_result),
                        "timestamp": datetime.now(UTC).isoformat()
                    })
                    
                    # Check if task is complete
                    task_complete = self._check_task_completion(step_plan, serialized_result, execution_state)
                    
                    return {
                        "status": "success",
                        "result": serialized_result,
                        "tool_name": tool_name,
                        "parameters": parameters,
                        "reasoning": step_plan["reasoning"],
                        "task_complete": task_complete,
                        "step_number": execution_state["current_step"]
                    }
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Log the specific error for debugging
                    print(f"🌐 ❌ Tool execution error: {error_msg}")
                    
                    # Check if it's a browser context error
                    if "Target page, context or browser has been closed" in error_msg:
                        retry_count += 1
                        if retry_count < max_retries:
                            log_step(f"🌐 🔄 Browser context error, retrying ({retry_count}/{max_retries})...", symbol="🌐")
                            # Wait a bit before retrying
                            await asyncio.sleep(1)
                            continue
                        else:
                            # Max retries reached
                            error_msg = f"Browser context error after {max_retries} retries: {error_msg}"
                    else:
                        # For other errors, don't retry
                        print(f"🌐 ❌ Non-retryable error: {error_msg}")
                    
                    # Store the error
                    execution_state["failed_steps"].append({
                        "status": "error",
                        "result": {"error": error_msg},
                        "step_number": execution_state["current_step"]
                    })
                    
                    return {
                        "status": "error",
                        "result": {"error": error_msg},
                        "step_number": execution_state["current_step"]
                    }
            
        except Exception as e:
            error_msg = f"Unexpected error in step {execution_state['current_step']}: {str(e)}"
            log_error(error_msg)
            
            execution_state["failed_steps"].append({
                "status": "error",
                "result": {"error": error_msg},
                "step_number": execution_state["current_step"]
            })
            
            return {
                "status": "error",
                "result": {"error": error_msg},
                "step_number": execution_state["current_step"]
            }
    
    def _get_available_tools(self) -> List[str]:
        """
        Get list of available browser tools.
        """
        if not self.multi_mcp:
            return []
        
        tools = self.multi_mcp.get_all_tools()
        return [tool.name for tool in tools]
    
    def _extract_url_from_result(self, result: Any) -> str:
        """
        Extract current URL from tool result.
        """
        if isinstance(result, dict):
            # Look for URL in common result patterns
            for key in ["url", "current_url", "page_url", "location"]:
                if key in result:
                    return str(result[key])
        
        # If no URL found, return empty string
        return ""
    
    def _serialize_result(self, result: Any) -> Any:
        """
        Convert result to JSON-serializable format.
        Handles CallToolResult objects and other non-serializable types.
        """
        try:
            # If it's already a basic type, return as is
            if isinstance(result, (str, int, float, bool, type(None))):
                return result
            
            # If it's a dict, recursively serialize its values
            if isinstance(result, dict):
                return {k: self._serialize_result(v) for k, v in result.items()}
            
            # If it's a list, recursively serialize its items
            if isinstance(result, list):
                return [self._serialize_result(item) for item in result]
            
            # For CallToolResult objects, try to extract content
            if hasattr(result, 'content') and hasattr(result.content, '__getitem__'):
                try:
                    content_text = result.content[0].text.strip()
                    # Try to parse as JSON
                    import json
                    parsed = json.loads(content_text)
                    return self._serialize_result(parsed)
                except (IndexError, AttributeError, json.JSONDecodeError):
                    # If JSON parsing fails, return the text content as string
                    try:
                        return result.content[0].text.strip()
                    except (IndexError, AttributeError):
                        pass
            
            # For other objects, convert to string representation
            return str(result)
            
        except Exception:
            # Fallback: convert to string
            return str(result)
    
    def _check_task_completion(self, step_plan: Dict[str, Any], result: Any, execution_state: Dict[str, Any]) -> bool:
        """
        Check if the current task is complete based on step plan and result.
        """
        # Check if the step plan indicates completion
        if step_plan.get("task_complete", False):
            return True
        
        # Check if we have a "done" action
        if step_plan.get("tool_name") == "done":
            return True
        
        # Check if we've reached a success condition
        if isinstance(result, dict) and result.get("success") is True:
            return True
        
        # Check if we have sufficient completed steps
        if len(execution_state["completed_steps"]) >= 5:
            return True
        
        return False

    async def _handle_form_interaction(self, form_data: Dict[str, str], execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle complex form interactions with retry logic.
        """
        log_step("📝 Handling form interaction", symbol="🌐")
        
        form_steps = []
        
        # Get interactive elements to find form fields
        try:
            elements_result = await self.multi_mcp.function_wrapper("get_interactive_elements", "visible", True)
            serialized_elements = self._serialize_result(elements_result)
            form_steps.append({
                "action": "get_interactive_elements",
                "result": serialized_elements,
                "status": "success"
            })
        except Exception as e:
            log_error("Failed to get interactive elements", e)
            return {"status": "error", "result": f"Form interaction failed: {str(e)}"}
        
        # Fill form fields
        for field_name, field_value in form_data.items():
            try:
                # Find field by name or type
                field_index = self._find_form_field_index(serialized_elements, field_name)
                if field_index is not None:
                    input_result = await self.multi_mcp.function_wrapper("input_text", field_index, field_value)
                    serialized_input = self._serialize_result(input_result)
                    form_steps.append({
                        "action": f"input_text_{field_name}",
                        "result": serialized_input,
                        "status": "success"
                    })
                else:
                    log_error(f"Could not find form field: {field_name}")
                    form_steps.append({
                        "action": f"input_text_{field_name}",
                        "result": f"Field not found: {field_name}",
                        "status": "error"
                    })
            except Exception as e:
                log_error(f"Failed to fill field {field_name}", e)
                form_steps.append({
                    "action": f"input_text_{field_name}",
                    "result": f"Error: {str(e)}",
                    "status": "error"
                })
        
        # Submit form
        try:
            submit_index = self._find_submit_button_index(serialized_elements)
            if submit_index is not None:
                submit_result = await self.multi_mcp.function_wrapper("click_element_by_index", submit_index)
                serialized_submit = self._serialize_result(submit_result)
                form_steps.append({
                    "action": "submit_form",
                    "result": serialized_submit,
                    "status": "success"
                })
            else:
                log_error("Could not find submit button")
                form_steps.append({
                    "action": "submit_form",
                    "result": "Submit button not found",
                    "status": "error"
                })
        except Exception as e:
            log_error("Failed to submit form", e)
            form_steps.append({
                "action": "submit_form",
                "result": f"Error: {str(e)}",
                "status": "error"
            })
        
        return {
            "status": "success",
            "result": form_steps,
            "form_completed": True
        }
    
    def _find_form_field_index(self, elements_result: Any, field_name: str) -> Optional[int]:
        """
        Find form field index by name or type.
        """
        if isinstance(elements_result, str):
            # Parse elements string to find field
            lines = elements_result.split('\n')
            for line in lines:
                if field_name.lower() in line.lower():
                    # Extract index from line like "[5]<input name='email'>"
                    import re
                    match = re.search(r'\[(\d+)\]', line)
                    if match:
                        return int(match.group(1))
        return None
    
    def _find_submit_button_index(self, elements_result: Any) -> Optional[int]:
        """
        Find submit button index.
        """
        if isinstance(elements_result, str):
            lines = elements_result.split('\n')
            for line in lines:
                # Look specifically for submit buttons
                if 'type="submit"' in line.lower() or 'type=\'submit\'' in line.lower():
                    import re
                    match = re.search(r'\[(\d+)\]', line)
                    if match:
                        return int(match.group(1))
        return None


def build_browser_agent_input(instruction: str, browser_state: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Build input for BrowserAgent from instruction.
    """
    return {
        "instruction": instruction,
        "browser_state": browser_state or {},
        "timestamp": datetime.now(UTC).isoformat()
    } 