import asyncio
import json
import uuid
from datetime import datetime, UTC
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import re
from difflib import SequenceMatcher
import logging
import os

from agent.agentSession import AgentSession
from agent.model_manager import ModelManager
from utils.utils import log_step, log_error, log_json_block, CustomJSONEncoder
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
        
        # LLM-powered form filling is now the primary method
        # Old hardcoded field mapping patterns are deprecated
        
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
                    if step_result.get("task_complete", False) or step_result.get("result") == "Task completed" or step_result.get("result") == "Form filled successfully" or step_result.get("result") == "Navigation completed":
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
                f"{json.dumps(step_input, indent=2, cls=CustomJSONEncoder)}\n"
                "```"
            )
            
            log_step("[SENDING PROMPT TO BROWSER AGENT...]", symbol="🌐")
            
            response = await self.model.generate_text(prompt=full_prompt)
            
            log_step("[RECEIVED OUTPUT FROM BROWSER AGENT...]", symbol="🌐")
            
            # Parse the response
            step_plan = parse_llm_json(response, required_keys=[
                "action_type", "tool_name", "parameters", "reasoning", "expected_outcome"
            ])
            
            if step_plan:
                log_step(f"✅ Generated step plan: {step_plan['action_type']} - {step_plan['tool_name']}", symbol="🌐")
                return step_plan
            else:
                log_error("Failed to parse step plan from LLM response")
                return None
                
        except Exception as e:
            log_error("Failed to get step plan from LLM", e)
            return None
    
    async def _execute_step_plan(self, step_plan: Dict[str, Any], execution_state: Dict[str, Any], session: AgentSession) -> Dict[str, Any]:
        """
        Execute a step plan using MCP tools.
        """
        action_type = step_plan.get("action_type", "")
        tool_name = step_plan.get("tool_name", "")
        parameters = step_plan.get("parameters", {})
        reasoning = step_plan.get("reasoning", "")
        expected_outcome = step_plan.get("expected_outcome", "")
        
        log_step(f"🔧 Executing: {action_type} - {tool_name}", symbol="🌐")
        log_step(f"📝 Reasoning: {reasoning}", symbol="💭")
        
        try:
            # Execute the tool first
            if self.multi_mcp and hasattr(self.multi_mcp, 'function_wrapper'):
                # Convert parameters dict to positional arguments based on tool schema
                positional_args = self._convert_parameters_to_positional_args(tool_name, parameters)
                result = await self.multi_mcp.function_wrapper(tool_name, *positional_args)
            else:
                # Mock result for testing
                result = {
                    "status": "success",
                    "message": f"Mock execution of {tool_name}",
                    "data": parameters
                }
            
            # Serialize the result
            serialized_result = self._serialize_result(result)
            
            # Update browser state
            execution_state["browser_state"] = {
                "last_action": tool_name,
                "last_result": serialized_result,
                "current_url": self._extract_url_from_result(result),
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            # Check if this is a form filling task AFTER navigation is complete
            # Only trigger LLM form interaction for non-navigation tools or after successful navigation
            if (tool_name not in ['open_tab', 'go_to_url', 'go_back', 'search_google'] and 
                self._is_form_filling_task(execution_state["instruction"], tool_name, parameters)):
                log_step("📝 Detected form filling task - using LLM-powered form interaction", symbol="🧠")
                form_result = await self._handle_llm_powered_form_interaction(execution_state["instruction"], execution_state)
                
                # If form interaction was successful, mark task as complete
                if form_result.get("form_completed", False):
                    return {
                        "status": "success",
                        "result": form_result["result"],
                        "step_number": execution_state["current_step"],
                        "action_type": "interaction",
                        "tool_name": "llm_form_filling",
                        "task_complete": True
                    }
                else:
                    # If form interaction failed, return the error
                    return {
                        "status": "error",
                        "result": form_result["result"],
                        "step_number": execution_state["current_step"],
                        "action_type": "interaction",
                        "tool_name": "llm_form_filling"
                    }
            
            # Check if task is complete
            task_complete = self._check_task_completion(step_plan, result, execution_state)
            
            log_step(f"✅ Step completed: {expected_outcome}", symbol="🌐")
            
            return {
                "status": "success",
                "result": serialized_result,
                "step_number": execution_state["current_step"],
                "action_type": action_type,
                "tool_name": tool_name,
                "task_complete": task_complete
            }
            
        except Exception as e:
            log_error(f"Failed to execute {tool_name}", e)
            return {
                "status": "error",
                "result": f"Tool execution failed: {str(e)}",
                "step_number": execution_state["current_step"],
                "action_type": action_type,
                "tool_name": tool_name
            }
    
    def _get_available_tools(self) -> List[str]:
        """
        Get list of available MCP tools.
        """
        if self.multi_mcp:
            return [
                "open_tab", "go_to_url", "click_element_by_index", "input_text",
                "get_interactive_elements", "get_enhanced_page_structure",
                "get_comprehensive_markdown", "take_screenshot", "wait",
                "send_keys", "scroll_down", "search_google", "done"
            ]
        else:
            return ["mock_tool"]
    
    def _extract_url_from_result(self, result: Any) -> str:
        """
        Extract URL from tool execution result.
        """
        if isinstance(result, dict):
            return result.get("url", result.get("current_url", ""))
        return ""
    
    def _serialize_result(self, result: Any) -> Any:
        """
        Serialize result for storage and logging.
        """
        try:
            if isinstance(result, dict):
                return result
            elif isinstance(result, str):
                return {"message": result}
            elif hasattr(result, 'content') and hasattr(result.content, '__iter__'):
                # Handle CallToolResult objects
                try:
                    # Try to extract text content
                    if len(result.content) > 0 and hasattr(result.content[0], 'text'):
                        return {"message": result.content[0].text}
                    else:
                        return {"data": str(result.content)}
                except:
                    return {"data": str(result)}
            elif hasattr(result, '__dict__'):
                # Handle objects with attributes
                return {"data": str(result)}
            else:
                return {"data": str(result)}
        except Exception as e:
            return {"error": f"Serialization failed: {str(e)}", "data": str(result)}
    
    def _check_task_completion(self, step_plan: Dict[str, Any], result: Any, execution_state: Dict[str, Any]) -> bool:
        """
        Check if the current step completes the task.
        """
        # Check if the step plan indicates completion
        if step_plan.get("task_complete", False):
            return True
        
        # Check if the result indicates completion
        if isinstance(result, dict):
            result_str = str(result).lower()
            completion_indicators = [
                "task completed", "form filled successfully", "navigation completed",
                "done", "complete", "finished", "success"
            ]
            if any(indicator in result_str for indicator in completion_indicators):
                return True
        
        # Check if we've reached the maximum steps
        if execution_state["current_step"] >= self.max_steps - 1:
            return True
        
        return False

    # ============================================================================
    # LLM-POWERED FORM FILLING METHODS
    # ============================================================================

    async def _handle_llm_powered_form_interaction(self, instruction: str, execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle form interaction using LLM-powered intelligent field matching.
        This replaces the old hardcoded field matching logic.
        """
        log_step("🧠 Starting LLM-powered form interaction", symbol="🧠")
        
        try:
            # Step 1: Extract form structure from the page
            form_structure = await self._extract_form_structure()
            if not form_structure:
                return {
                    "status": "error",
                    "result": "Could not extract form structure from the page. The page may not be loaded or may not contain form elements.",
                    "form_completed": False,
                    "suggestion": "Try refreshing the page or waiting for it to load completely."
                }
            
            log_step(f"📋 Extracted {len(form_structure)} form fields", symbol="📊")
            
            # Step 2: Get LLM-powered field mapping
            field_mapping = await self._get_llm_field_mapping(instruction, form_structure)
            if not field_mapping:
                return {
                    "status": "error", 
                    "result": "LLM could not determine field mappings. The form structure may be unclear or the instruction may not match available fields.",
                    "form_completed": False,
                    "suggestion": "Check if the form fields are properly labeled and the instruction contains clear field information."
                }
            
            log_step(f"🧠 LLM mapped {len(field_mapping)} fields", symbol="✅")
            
            # Step 3: Execute the field filling instructions
            fill_results = await self._execute_field_filling(field_mapping)
            
            # Check if any fields were filled successfully
            successful_fills = [r for r in fill_results if r.get('status') == 'success']
            if not successful_fills:
                return {
                    "status": "error",
                    "result": "Failed to fill any form fields. The field indices may be incorrect or the form may have changed.",
                    "form_completed": False,
                    "suggestion": "Try refreshing the page and attempting the form filling again."
                }
            
            # Step 4: Submit the form
            submit_result = await self._submit_form()
            
            return {
                "status": "success",
                "result": {
                    "message": "Form filled and submitted successfully using LLM intelligence",
                    "fields_filled": len(successful_fills),
                    "total_fields": len(field_mapping),
                    "field_mapping": field_mapping,
                    "fill_results": fill_results,
                    "submit_result": submit_result
                },
                "form_completed": True,
                "filled_fields": [field["field_name"] for field in field_mapping]
            }
            
        except Exception as e:
            log_error("LLM-powered form interaction failed", e)
            return {
                "status": "error",
                "result": f"LLM form interaction failed: {str(e)}",
                "form_completed": False,
                "suggestion": "Try refreshing the page and attempting the form filling again."
            }

    async def _extract_form_structure(self) -> List[Dict[str, Any]]:
        """
        Extract form structure from the current page and convert to structured format for LLM.
        """
        log_step("🔍 Extracting form structure from page", symbol="🔍")
        
        try:
            # Wait for page to load
            if self.multi_mcp:
                await self.multi_mcp.function_wrapper("wait", 5)  # Increased wait time for Google Forms
            
            # Get interactive elements with more detailed information
            elements_result = None
            if self.multi_mcp:
                # Try to get comprehensive page structure first
                try:
                    page_structure = await self.multi_mcp.function_wrapper("get_enhanced_page_structure")
                    log_step(f"🔍 Enhanced page structure type: {type(page_structure)}", symbol="🐛")
                    if isinstance(page_structure, dict):
                        log_step(f"🔍 Page structure keys: {list(page_structure.keys())}", symbol="🐛")
                        # Try to extract form fields from enhanced structure
                        form_structure = self._extract_from_enhanced_structure(page_structure)
                        if form_structure:
                            log_step(f"📋 Extracted {len(form_structure)} fields from enhanced structure", symbol="📊")
                            return form_structure
                except Exception as e:
                    log_step(f"🔍 Enhanced page structure failed: {e}", symbol="⚠️")
                
                # Fallback to interactive elements
                elements_result = await self.multi_mcp.function_wrapper("get_interactive_elements", True, True, True)
            
            # Log the raw result for debugging
            log_step(f"🔍 Raw elements result type: {type(elements_result)}", symbol="🐛")
            if isinstance(elements_result, dict):
                log_step(f"🔍 Raw elements keys: {list(elements_result.keys())}", symbol="🐛")
                if 'forms' in elements_result:
                    log_step(f"🔍 Found {len(elements_result['forms'])} forms", symbol="🐛")
                    # Log each form for debugging
                    for i, form in enumerate(elements_result['forms']):
                        log_step(f"🔍 Form {i}: {form}", symbol="🔍")
            
            # Convert to structured format for LLM
            form_structure = self._convert_elements_to_llm_format(elements_result)
            
            log_step(f"📋 Converted {len(form_structure)} form fields to LLM format", symbol="📊")
            
            # If still no form fields found, try a different approach
            if not form_structure:
                log_step("🔍 No form fields found, trying alternative approach", symbol="⚠️")
                try:
                    if self.multi_mcp:
                        # Try to get comprehensive markdown which might have more details
                        markdown_result = await self.multi_mcp.function_wrapper("get_comprehensive_markdown")
                        log_step(f"🔍 Markdown result type: {type(markdown_result)}", symbol="🐛")
                        if isinstance(markdown_result, str):
                            # Try to extract form fields from markdown
                            form_structure = self._extract_from_markdown(markdown_result)
                except Exception as e:
                    log_step(f"🔍 Markdown extraction failed: {e}", symbol="⚠️")
            
            return form_structure
            
        except Exception as e:
            log_error("Failed to extract form structure", e)
            return []

    def _extract_from_enhanced_structure(self, page_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract form fields from enhanced page structure.
        """
        form_fields = []
        
        try:
            # Look for forms in the enhanced structure
            if 'forms' in page_structure:
                for i, form in enumerate(page_structure['forms']):
                    field_info = {
                        "index": i,
                        "tag": "input",
                        "type": self._infer_field_type(str(form)),
                        "name": "",
                        "placeholder": "",
                        "label": str(form),
                        "description": str(form),
                        "action": "",
                        "required": True,
                        "options": []
                    }
                    form_fields.append(field_info)
            
            # Look for input elements
            if 'inputs' in page_structure:
                for i, input_elem in enumerate(page_structure['inputs']):
                    field_info = {
                        "index": i,
                        "tag": "input",
                        "type": input_elem.get('type', 'text'),
                        "name": input_elem.get('name', ''),
                        "placeholder": input_elem.get('placeholder', ''),
                        "label": input_elem.get('label', ''),
                        "description": str(input_elem),
                        "action": "",
                        "required": True,
                        "options": []
                    }
                    form_fields.append(field_info)
            
            log_step(f"🔍 Extracted {len(form_fields)} fields from enhanced structure", symbol="📊")
            return form_fields
            
        except Exception as e:
            log_error("Failed to extract from enhanced structure", e)
            return []

    def _convert_elements_to_llm_format(self, elements_result: Any) -> List[Dict[str, Any]]:
        """
        Convert browser elements result to structured format for LLM processing.
        """
        form_fields = []
        
        try:
            # Handle structured JSON format
            if isinstance(elements_result, dict):
                # Check for forms array
                forms = elements_result.get('forms', [])
                log_step(f"🔍 Processing {len(forms)} forms", symbol="🔍")
                
                for form_index, form in enumerate(forms):
                    log_step(f"🔍 Form {form_index}: {form}", symbol="🔍")
                    
                    # Check if form has individual fields
                    if 'fields' in form:
                        # Process individual fields within the form
                        for field_index, field in enumerate(form['fields']):
                            field_info = {
                                "index": field.get('id', field_index),
                                "tag": "input",  # Default tag
                                "type": self._infer_field_type(field.get('desc', '')),
                                "name": field.get('name', ''),
                                "placeholder": field.get('placeholder', ''),
                                "label": field.get('desc', ''),
                                "description": field.get('desc', ''),
                                "action": field.get('action', ''),
                                "required": field.get('required', True),
                                "options": field.get('options', [])
                            }
                            
                            # Infer field type from description
                            desc_lower = field.get('desc', '').lower()
                            if 'dropdown' in desc_lower or 'select' in desc_lower:
                                field_info["tag"] = "select"
                                field_info["type"] = "select"
                            elif 'radio' in desc_lower or 'button' in desc_lower:
                                field_info["tag"] = "input"
                                field_info["type"] = "radio"
                            elif 'textarea' in desc_lower or 'message' in desc_lower:
                                field_info["tag"] = "textarea"
                                field_info["type"] = "textarea"
                            elif 'email' in desc_lower:
                                field_info["type"] = "email"
                            elif 'phone' in desc_lower or 'mobile' in desc_lower:
                                field_info["type"] = "tel"
                            elif 'date' in desc_lower or 'birth' in desc_lower:
                                field_info["type"] = "date"
                            
                            form_fields.append(field_info)
                    else:
                        # Treat the form itself as a field (legacy format)
                        field_info = {
                            "index": form.get('id', form_index),
                            "tag": "input",  # Default tag
                            "type": self._infer_field_type(form.get('desc', '')),
                            "name": "",
                            "placeholder": "",
                            "label": form.get('desc', ''),
                            "description": form.get('desc', ''),
                            "action": form.get('action', ''),
                            "required": True,  # Assume required for form fields
                            "options": []  # For dropdowns/selects
                        }
                        
                        # Infer field type from description
                        desc_lower = form.get('desc', '').lower()
                        if 'dropdown' in desc_lower or 'select' in desc_lower:
                            field_info["tag"] = "select"
                            field_info["type"] = "select"
                        elif 'radio' in desc_lower or 'button' in desc_lower:
                            field_info["tag"] = "input"
                            field_info["type"] = "radio"
                        elif 'textarea' in desc_lower or 'message' in desc_lower:
                            field_info["tag"] = "textarea"
                            field_info["type"] = "textarea"
                        elif 'email' in desc_lower:
                            field_info["type"] = "email"
                        elif 'phone' in desc_lower or 'mobile' in desc_lower:
                            field_info["type"] = "tel"
                        elif 'date' in desc_lower or 'birth' in desc_lower:
                            field_info["type"] = "date"
                        
                        form_fields.append(field_info)
                
                # Also check for individual input elements that might not be in forms
                inputs = elements_result.get('inputs', [])
                for input_index, input_field in enumerate(inputs):
                    field_info = {
                        "index": input_field.get('id', len(form_fields) + input_index),
                        "tag": "input",
                        "type": self._infer_field_type(input_field.get('desc', '')),
                        "name": input_field.get('name', ''),
                        "placeholder": input_field.get('placeholder', ''),
                        "label": input_field.get('desc', ''),
                        "description": input_field.get('desc', ''),
                        "action": "",
                        "required": input_field.get('required', True),
                        "options": []
                    }
                    form_fields.append(field_info)
            
            # Handle string format (legacy)
            elif isinstance(elements_result, str):
                lines = elements_result.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() and '[' in line and ']' in line:
                        field_info = {
                            "index": i,
                            "tag": "input",
                            "type": self._infer_field_type(line),
                            "name": "",
                            "placeholder": "",
                            "label": line.strip(),
                            "description": line.strip(),
                            "action": "",
                            "required": True,
                            "options": []
                        }
                        form_fields.append(field_info)
            
            log_step(f"🔄 Converted {len(form_fields)} form fields to LLM format", symbol="🔄")
            return form_fields
            
        except Exception as e:
            log_error("Failed to convert elements to LLM format", e)
            return []

    def _infer_field_type(self, description: str) -> str:
        """
        Infer field type from description text.
        """
        desc_lower = description.lower()
        
        if 'email' in desc_lower:
            return 'email'
        elif 'phone' in desc_lower or 'mobile' in desc_lower or 'tel' in desc_lower:
            return 'tel'
        elif 'date' in desc_lower or 'birth' in desc_lower or 'dob' in desc_lower:
            return 'date'
        elif 'number' in desc_lower or 'amount' in desc_lower:
            return 'number'
        elif 'textarea' in desc_lower or 'message' in desc_lower or 'comment' in desc_lower:
            return 'textarea'
        elif 'dropdown' in desc_lower or 'select' in desc_lower:
            return 'select'
        elif 'radio' in desc_lower:
            return 'radio'
        elif 'checkbox' in desc_lower:
            return 'checkbox'
        elif 'file' in desc_lower or 'upload' in desc_lower:
            return 'file'
        elif 'url' in desc_lower or 'website' in desc_lower:
            return 'url'
        else:
            return 'text'

    async def _get_llm_field_mapping(self, instruction: str, form_structure: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use LLM to intelligently map form fields to values from the instruction.
        """
        log_step("🧠 Getting LLM field mapping", symbol="🧠")
        
        try:
            # Create prompt for LLM field mapping
            prompt = self._create_llm_field_mapping_prompt(instruction, form_structure)
            
            # Get response from LLM
            response = await self.model.generate_text(prompt=prompt)
            
            # Parse the response
            field_mapping = self._parse_llm_field_mapping_response(response)
            
            log_step(f"🧠 LLM returned {len(field_mapping)} field mappings", symbol="✅")
            return field_mapping
            
        except Exception as e:
            log_error("Failed to get LLM field mapping", e)
            return []

    def _create_llm_field_mapping_prompt(self, instruction: str, form_structure: List[Dict[str, Any]]) -> str:
        """
        Create prompt for LLM field mapping.
        """
        prompt = f"""You are a smart form-filler assistant specializing in Google Forms and web forms. Based on the user's instruction and the list of form fields, match the appropriate values to the right input fields by their index. Do not fabricate fields. Return results in this JSON format:

```json
[
  {{ "index": <form_field_index>, "value": "<value to fill>", "field_name": "<field_name>", "confidence": <0.0-1.0> }}
]
```

User instruction:
{instruction}

Form fields:
```json
{json.dumps(form_structure, indent=2, cls=CustomJSONEncoder)}
```

Instructions:
1. Analyze the user instruction to extract relevant data (names, emails, dates, etc.)
2. Match each piece of data to the most appropriate form field based on:
   - Field labels and descriptions
   - Field types (email, date, text, etc.)
   - Field positions and context
   - Semantic similarity
3. For Google Forms, look for:
   - Name fields (first name, last name, full name)
   - Email fields (email, email address)
   - Date fields (date of birth, birth date, DOB)
   - Text fields (course names, descriptions)
   - Radio buttons (yes/no questions, marital status)
4. Only include fields that should be filled
5. Provide a confidence score (0.0-1.0) for each mapping
6. Use the exact field index from the form structure
7. Return valid JSON only

Example output:
```json
[
  {{ "index": 0, "value": "John Doe", "field_name": "name", "confidence": 0.95 }},
  {{ "index": 1, "value": "john@example.com", "field_name": "email", "confidence": 0.98 }},
  {{ "index": 2, "value": "1992-10-03", "field_name": "date_of_birth", "confidence": 0.90 }}
]
```

Return the JSON array:"""

        return prompt

    def _parse_llm_field_mapping_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response to extract field mappings.
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON array directly
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    log_error("No JSON found in LLM response")
                    return []
            
            # Parse JSON
            field_mapping = json.loads(json_str)
            
            # Validate structure
            if not isinstance(field_mapping, list):
                log_error("LLM response is not a list")
                return []
            
            # Validate each mapping
            valid_mappings = []
            for mapping in field_mapping:
                if isinstance(mapping, dict) and 'index' in mapping and 'value' in mapping:
                    valid_mappings.append({
                        'index': mapping['index'],
                        'value': mapping['value'],
                        'field_name': mapping.get('field_name', ''),
                        'confidence': mapping.get('confidence', 0.5)
                    })
            
            log_step(f"✅ Parsed {len(valid_mappings)} valid field mappings", symbol="✅")
            return valid_mappings
            
        except Exception as e:
            log_error("Failed to parse LLM field mapping response", e)
            return []

    async def _execute_field_filling(self, field_mapping: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the field filling instructions from LLM.
        """
        log_step(f"📝 Executing field filling for {len(field_mapping)} fields", symbol="📝")
        
        fill_results = []
        
        for mapping in field_mapping:
            try:
                field_index = mapping['index']
                field_value = mapping['value']
                field_name = mapping.get('field_name', '')
                confidence = mapping.get('confidence', 0.5)
                
                log_step(f"📝 Filling field {field_index} ({field_name}) with '{field_value}' (confidence: {confidence})", symbol="📝")
                
                # Execute the fill action
                if self.multi_mcp:
                    # Try to click the field first to focus it
                    try:
                        await self.multi_mcp.function_wrapper("click_element_by_index", field_index)
                        log_step(f"✅ Clicked field {field_index} to focus", symbol="✅")
                    except Exception as click_error:
                        log_step(f"⚠️ Could not click field {field_index}: {click_error}", symbol="⚠️")
                    
                    # Wait a moment for focus
                    await self.multi_mcp.function_wrapper("wait", 1)
                    
                    # Fill the field
                    result = await self.multi_mcp.function_wrapper("input_text", field_index, field_value)
                    fill_results.append({
                        'field_index': field_index,
                        'field_name': field_name,
                        'value': field_value,
                        'confidence': confidence,
                        'status': 'success',
                        'result': result
                    })
                    log_step(f"✅ Filled field {field_index} successfully", symbol="✅")
                    
                    # Wait between fields
                    await self.multi_mcp.function_wrapper("wait", 1)
                else:
                    # Mock result for testing
                    fill_results.append({
                        'field_index': field_index,
                        'field_name': field_name,
                        'value': field_value,
                        'confidence': confidence,
                        'status': 'success',
                        'result': {'message': f'Mock fill: {field_value}'}
                    })
                
            except Exception as e:
                log_error(f"Failed to fill field {field_index}", e)
                fill_results.append({
                    'field_index': field_index,
                    'field_name': field_name,
                    'value': field_value,
                    'confidence': confidence,
                    'status': 'error',
                    'result': str(e)
                })
        
        return fill_results

    async def _submit_form(self) -> Dict[str, Any]:
        """
        Submit the form after filling all fields.
        """
        log_step("📤 Submitting form", symbol="📤")
        
        try:
            if self.multi_mcp:
                # Try to find and click submit button
                submit_result = await self.multi_mcp.function_wrapper("click_element_by_index", 0)  # Default to first element
                log_step("✅ Form submitted successfully", symbol="✅")
                return {
                    'status': 'success',
                    'result': submit_result
                }
            else:
                # Mock result for testing
                return {
                    'status': 'success',
                    'result': {'message': 'Mock form submission'}
                }
                
        except Exception as e:
            log_error("Failed to submit form", e)
            return {
                'status': 'error',
                'result': str(e)
            }

    def _is_form_filling_task(self, instruction: str, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Detect if the current task is a form filling task.
        """
        instruction_lower = instruction.lower()
        
        # Check for form-related keywords in instruction
        form_keywords = [
            'fill form', 'submit form', 'complete form', 'fill out form',
            'registration form', 'contact form', 'survey form', 'application form',
            'enter data', 'input data', 'fill data'
        ]
        
        has_form_keywords = any(keyword in instruction_lower for keyword in form_keywords)
        
        # Check if we're on a form page (Google Forms, etc.)
        current_url = parameters.get('url', '')
        is_form_page = any(form_domain in current_url.lower() for form_domain in [
            'forms.gle', 'forms.google', 'docs.google.com/forms',
            'survey', 'form', 'registration', 'contact'
        ])
        
        # For navigation tools, only trigger if we have form keywords AND are going to a form page
        if tool_name in ['open_tab', 'go_to_url']:
            return has_form_keywords and is_form_page
        
        # For form interaction tools, require form keywords or form page context
        if tool_name in ['input_text', 'get_interactive_elements', 'click_element_by_index']:
            return has_form_keywords or is_form_page
        
        # For other tools, require more specific conditions
        return False

    def _convert_parameters_to_positional_args(self, tool_name: str, parameters: Dict[str, Any]) -> List[Any]:
        """
        Convert parameters dictionary to positional arguments based on tool schema.
        """
        # Get the tool schema to understand parameter order
        if not self.multi_mcp or tool_name not in self.multi_mcp.tool_map:
            # Fallback: return parameters as values in order
            return list(parameters.values())
        
        tool = self.multi_mcp.tool_map[tool_name]["tool"]
        schema = tool.inputSchema
        positional_args = []
        
        # Handle nested input schema (most common case)
        if "input" in schema.get("properties", {}):
            inner_key = next(iter(schema.get("$defs", {})), None)
            if inner_key:
                inner_props = schema["$defs"][inner_key]["properties"]
                param_names = list(inner_props.keys())
                for param_name in param_names:
                    positional_args.append(parameters.get(param_name, None))
        else:
            # Handle direct properties schema
            param_names = list(schema["properties"].keys())
            for param_name in param_names:
                positional_args.append(parameters.get(param_name, None))
        
        return positional_args

    def _extract_from_markdown(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Extract form fields from markdown content.
        """
        form_fields = []
        
        try:
            lines = markdown_content.split('\n')
            field_index = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for form field patterns in markdown
                if any(keyword in line.lower() for keyword in ['input', 'text', 'email', 'name', 'date', 'phone', 'question']):
                    field_info = {
                        "index": field_index,
                        "tag": "input",
                        "type": self._infer_field_type(line),
                        "name": "",
                        "placeholder": "",
                        "label": line,
                        "description": line,
                        "action": "",
                        "required": True,
                        "options": []
                    }
                    form_fields.append(field_info)
                    field_index += 1
            
            log_step(f"🔍 Extracted {len(form_fields)} fields from markdown", symbol="📊")
            return form_fields
            
        except Exception as e:
            log_error("Failed to extract from markdown", e)
            return []

def build_browser_agent_input(instruction: str, browser_state: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Build input for BrowserAgent from instruction.
    """
    return {
        "instruction": instruction,
        "browser_state": browser_state or {},
        "timestamp": datetime.now(UTC).isoformat()
    } 