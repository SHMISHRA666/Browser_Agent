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
from dateutil.parser import parse

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
        This replaces the old hardcoded field matching logic with enhanced error handling.
        """
        log_step("🧠 Starting LLM-powered form interaction", symbol="🧠")
        
        try:
            # Step 1: Extract form structure from the page
            log_step("📋 Step 1: Extracting form structure", symbol="1️⃣")
            form_structure = await self._extract_form_structure()
            if not form_structure:
                log_step("❌ No form structure found", symbol="❌")
                return {
                    "status": "error",
                    "result": "Could not extract form structure from the page. The page may not be loaded or may not contain form elements.",
                    "form_completed": False,
                    "suggestion": "Try refreshing the page or waiting for it to load completely.",
                    "debug_info": {
                        "step": "form_structure_extraction",
                        "error": "No form fields detected"
                    }
                }
            
            log_step(f"📋 Extracted {len(form_structure)} form fields", symbol="📊")
            
            # Log form structure for debugging
            for i, field in enumerate(form_structure):
                log_step(f"📋 Field {i}: {field.get('type', 'unknown')} - {field.get('label', 'no label')} - Options: {field.get('options', [])}", symbol="🔍")
            
            # Step 2: Get LLM-powered field mapping
            log_step("🧠 Step 2: Getting LLM field mapping", symbol="2️⃣")
            field_mapping = await self._get_llm_field_mapping(instruction, form_structure)
            if not field_mapping:
                log_step("❌ No field mapping generated", symbol="❌")
                return {
                    "status": "error", 
                    "result": "LLM could not determine field mappings. The form structure may be unclear or the instruction may not match available fields.",
                    "form_completed": False,
                    "suggestion": "Check if the form fields are properly labeled and the instruction contains clear field information.",
                    "debug_info": {
                        "step": "llm_field_mapping",
                        "error": "No field mappings returned",
                        "form_structure_count": len(form_structure)
                    }
                }
            
            log_step(f"🧠 LLM mapped {len(field_mapping)} fields", symbol="✅")
            
            # Log field mappings for debugging
            for mapping in field_mapping:
                log_step(f"🧠 Map: {mapping.get('field_name', 'unknown')} -> '{mapping.get('value', '')}' (confidence: {mapping.get('confidence', 0)})", symbol="🔗")
            
            # Step 3: Execute the field filling instructions
            log_step("📝 Step 3: Executing field filling", symbol="3️⃣")
            fill_results = await self._execute_field_filling(field_mapping)
            
            # Check if any fields were filled successfully
            successful_fills = [r for r in fill_results if r.get('status') == 'success']
            failed_fills = [r for r in fill_results if r.get('status') == 'error']
            
            log_step(f"📝 Field filling results: {len(successful_fills)} successful, {len(failed_fills)} failed", symbol="📊")
            
            if not successful_fills:
                log_step("❌ No fields filled successfully", symbol="❌")
                return {
                    "status": "error",
                    "result": "Failed to fill any form fields. The field indices may be incorrect or the form may have changed.",
                    "form_completed": False,
                    "suggestion": "Try refreshing the page and attempting the form filling again.",
                    "debug_info": {
                        "step": "field_filling",
                        "error": "No successful field fills",
                        "failed_fills": failed_fills
                    }
                }
            
            # Step 4: Submit the form
            log_step("📤 Step 4: Submitting form", symbol="4️⃣")
            submit_result = await self._submit_form()
            
            # Prepare detailed result
            result = {
                "status": "success",
                "result": {
                    "message": "Form filled and submitted successfully using LLM intelligence",
                    "fields_filled": len(successful_fills),
                    "total_fields": len(field_mapping),
                    "failed_fields": len(failed_fills),
                    "field_mapping": field_mapping,
                    "fill_results": fill_results,
                    "submit_result": submit_result
                },
                "form_completed": True,
                "filled_fields": [field["field_name"] for field in field_mapping],
                "debug_info": {
                    "form_structure_count": len(form_structure),
                    "field_mapping_count": len(field_mapping),
                    "successful_fills": len(successful_fills),
                    "failed_fills": len(failed_fills)
                }
            }
            
            log_step(f"✅ Form interaction completed successfully: {len(successful_fills)}/{len(field_mapping)} fields filled", symbol="✅")
            return result
            
        except Exception as e:
            log_error("LLM-powered form interaction failed", e)
            return {
                "status": "error",
                "result": f"LLM form interaction failed: {str(e)}",
                "form_completed": False,
                "suggestion": "Try refreshing the page and attempting the form filling again.",
                "debug_info": {
                    "step": "general_error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            }

    async def _extract_form_structure(self) -> List[Dict[str, Any]]:
        """
        Extract form structure from the current page and convert to structured format for LLM.
        Enhanced to detect complex form fields and provide detailed metadata.
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
                            # Check if the extracted fields have meaningful question text
                            meaningful_fields = [f for f in form_structure if f.get('question_text', '').lower() not in ['id', 'text input', 'input', 'field'] and not f.get('question_text', '').startswith('Form field')]
                            if len(meaningful_fields) < len(form_structure):
                                log_step("⚠️ Some fields lack meaningful question text, trying detailed analysis", symbol="⚠️")
                                try:
                                    detailed_form_structure = await self._get_detailed_form_info()
                                    if detailed_form_structure:
                                        log_step(f"📋 Detailed analysis found {len(detailed_form_structure)} fields", symbol="📊")
                                        return detailed_form_structure
                                except Exception as e:
                                    log_step(f"⚠️ Detailed analysis failed: {e}", symbol="⚠️")
                            return form_structure
                except Exception as e:
                    log_step(f"🔍 Enhanced page structure failed: {e}", symbol="⚠️")
                
                # Fallback to interactive elements with structured output
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
            
            # If no forms found, try Google Forms specific extraction
            if not form_structure:
                log_step("🔍 No traditional forms found, trying Google Forms specific extraction", symbol="⚠️")
                try:
                    google_forms_structure = await self._extract_google_forms_structure(elements_result)
                    if google_forms_structure:
                        log_step(f"📋 Google Forms extraction found {len(google_forms_structure)} fields", symbol="📊")
                        return google_forms_structure
                except Exception as e:
                    log_step(f"🔍 Google Forms extraction failed: {e}", symbol="⚠️")
            
            # Check if the extracted fields have meaningful question text
            meaningful_fields = [f for f in form_structure if f.get('question_text', '').lower() not in ['id', 'text input', 'input', 'field'] and not f.get('question_text', '').startswith('Form field')]
            if len(meaningful_fields) < len(form_structure):
                log_step("⚠️ Some fields lack meaningful question text, trying detailed analysis", symbol="⚠️")
                try:
                    # Try detailed form analysis as fallback
                    detailed_form_structure = await self._get_detailed_form_info()
                    if detailed_form_structure:
                        log_step(f"📋 Detailed analysis found {len(detailed_form_structure)} fields", symbol="📊")
                        return detailed_form_structure
                except Exception as e:
                    log_step(f"⚠️ Detailed analysis failed: {e}", symbol="⚠️")
            
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

    async def _extract_google_forms_structure(self, elements_result: Any) -> List[Dict[str, Any]]:
        """
        Extract form structure specifically for Google Forms by looking at all interactive elements.
        Google Forms doesn't use traditional HTML forms, so we need to look at inputs, textareas, etc.
        """
        form_fields = []
        
        try:
            if not isinstance(elements_result, dict):
                return []
            
            # Look for all possible interactive elements that could be form fields
            interactive_elements = []
            
            # Check for inputs
            if 'inputs' in elements_result:
                interactive_elements.extend(elements_result['inputs'])
            
            # Check for textareas
            if 'textareas' in elements_result:
                interactive_elements.extend(elements_result['textareas'])
            
            # Check for selects (dropdowns)
            if 'selects' in elements_result:
                interactive_elements.extend(elements_result['selects'])
            
            # Check for buttons that might be radio/checkbox groups
            if 'buttons' in elements_result:
                # Filter buttons that might be form-related
                form_buttons = []
                for btn in elements_result['buttons']:
                    btn_text = btn.get('text', '').lower()
                    # Skip submit buttons
                    if btn_text not in ['submit', 'send', 'next', 'done']:
                        form_buttons.append(btn)
                interactive_elements.extend(form_buttons)
            
            # Check for any other interactive elements
            for key, value in elements_result.items():
                if key not in ['success', 'nav', 'forms', 'total'] and isinstance(value, list):
                    interactive_elements.extend(value)
            
            log_step(f"🔍 Found {len(interactive_elements)} potential interactive elements", symbol="🔍")
            
            # Process each interactive element as a potential form field
            for i, element in enumerate(interactive_elements):
                if not isinstance(element, dict):
                    continue
                
                # Extract raw HTML and question text using the new function
                raw_html = element.get('raw_html', '')
                question_text = self._extract_question_from_html(raw_html)
                
                if not question_text:
                    question_text = f"Form field {element.get('id', i)}"
                
                # Assign correct type based on description
                desc = element.get("desc", "").lower()
                if "text" in desc:
                    field_type = "text"
                elif "radio" in desc:
                    field_type = "radio"
                elif "check" in desc:
                    field_type = "checkbox"
                elif "drop" in desc:
                    field_type = "dropdown"
                else:
                    field_type = "text"  # default fallback
                
                # Construct field with proper structure
                field = {
                    "index": i,
                    "id": element.get("id", i),
                    "question_text": question_text,
                    "type": field_type,
                    "options": element.get("options", []),
                    "raw_html": raw_html,
                    "desc": desc,
                    "action": element.get("action", ""),
                    "required": element.get("required", True),
                    "tag": self._get_field_tag(field_type),
                    "name": element.get("name", ""),
                    "placeholder": element.get("placeholder", ""),
                    "label": desc,
                    "description": desc
                }
                
                # Add debug logging for each field
                log_step(f"🧪 Field {i} -> '{question_text}' [type: {field_type}]", symbol="🔍")
                
                form_fields.append(field)
            
            log_step(f"📋 Extracted {len(form_fields)} fields from Google Forms structure", symbol="📊")
            # Extra logging for debugging field extraction
            for i, f in enumerate(form_fields):
                log_step(f"🧪 Field {i} debug context:\n"
                        f"  Question: {f.get('question_text')}\n"
                        f"  Raw HTML snippet: {f.get('raw_html', '')[:300]}", symbol="🧬")

            return form_fields
            
        except Exception as e:
            log_error("Failed to extract Google Forms structure", e)
            return []

    def _extract_question_from_html(self, raw_html: str) -> str:
        """
        Extract question text from HTML content using regex patterns.
        Find likely label strings from HTML content that aren't form controls or generic labels.
        """
        try:
            if not raw_html:
                return ""
            
            # Find likely label strings from HTML content
            matches = re.findall(r'<div[^>]*>(.*?)</div>', raw_html)
            texts = [m.strip() for m in matches if 5 < len(m.strip()) < 100]
            
            # Return first candidate that isn't a form control or generic label
            for txt in texts:
                if txt.lower() not in ['text input', 'field', 'input'] and not txt.lower().startswith('form field'):
                    return txt
            
            # If no div matches found, try other patterns
            # Look for aria-label, placeholder, or title attributes
            aria_patterns = [
                r'aria-label[=:]\s*["\']([^"\']+)["\']',
                r'placeholder[=:]\s*["\']([^"\']+)["\']',
                r'title[=:]\s*["\']([^"\']+)["\']'
            ]
            
            for pattern in aria_patterns:
                matches = re.findall(pattern, raw_html, re.IGNORECASE)
                for match in matches:
                    if match and match.lower() not in ['text input', 'input', 'field', 'id']:
                        return match.strip()
            
            # Try to extract from any text content
            text_match = re.search(r'>([^<>]{4,100})<', raw_html)
            if text_match:
                potential = text_match.group(1).strip()
                if not potential.lower().startswith("form field"):
                    return potential
            
            return ""
            
        except Exception as e:
            log_error("Failed to extract question from HTML", e)
            return ""

    def _extract_google_forms_question_text(self, element: Dict[str, Any]) -> str:
        """
        Extract question text from Google Forms elements using extended strategies.
        """
        try:
            # Priority 1: Use aria-label or title
            for attr in ['aria-label', 'ariaLabel', 'title', 'placeholder', 'text']:
                if attr in element and element[attr]:
                    val = element[attr].strip()
                    if val.lower() not in ['text input', 'input', 'field', 'id', 'form field']:
                        return val

            # Priority 2: Use parent_text if available
            if 'parent_text' in element:
                val = element['parent_text'].strip()
                if val.lower() not in ['text input', 'input', 'field', 'id', 'form field']:
                    return val

            # Priority 3: Extract from raw HTML using the new function
            raw_html = element.get("raw_html", "")
            question_text = self._extract_question_from_html(raw_html)
            if question_text:
                return question_text

            # Priority 4: Fallback to any other informative string
            for key, val in element.items():
                if isinstance(val, str) and 4 < len(val) < 100:
                    if any(w in val.lower() for w in ['name', 'email', 'course', 'married', 'date']):
                        return val.strip()

            return f"Form field {element.get('id', 'unknown')}"
        except Exception as e:
            log_error("Failed to extract Google Forms question text", e)
            return f"Form field {element.get('id', 'unknown')}"

    def _extract_from_enhanced_structure(self, page_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract form fields from enhanced page structure with improved field detection.
        """
        form_fields = []
        
        try:
            # Look for forms in the enhanced structure
            if 'forms' in page_structure:
                for i, form in enumerate(page_structure['forms']):
                    # Extract field information with better type detection
                    field_info = self._extract_enhanced_field_info(form, i)
                    if field_info:
                        form_fields.append(field_info)
            
            # Look for input elements
            if 'inputs' in page_structure:
                for i, input_elem in enumerate(page_structure['inputs']):
                    field_info = self._extract_enhanced_field_info(input_elem, i)
                    if field_info:
                        form_fields.append(field_info)
            
            # Look for select elements (dropdowns)
            if 'selects' in page_structure:
                for i, select_elem in enumerate(page_structure['selects']):
                    field_info = self._extract_enhanced_field_info(select_elem, i, field_type="dropdown")
                    if field_info:
                        form_fields.append(field_info)
            
            # Look for radio button groups
            if 'radio_groups' in page_structure:
                for i, radio_group in enumerate(page_structure['radio_groups']):
                    field_info = self._extract_enhanced_field_info(radio_group, i, field_type="radio")
                    if field_info:
                        form_fields.append(field_info)
            
            # Look for checkbox groups
            if 'checkbox_groups' in page_structure:
                for i, checkbox_group in enumerate(page_structure['checkbox_groups']):
                    field_info = self._extract_enhanced_field_info(checkbox_group, i, field_type="checkbox")
                    if field_info:
                        form_fields.append(field_info)
            
            log_step(f"🔍 Extracted {len(form_fields)} fields from enhanced structure", symbol="📊")
            return form_fields
            
        except Exception as e:
            log_error("Failed to extract from enhanced structure", e)
            return []

    def _extract_enhanced_field_info(self, field_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Extract enhanced field information with better question text detection and type inference.
        """
        try:
            # Get the basic field data
            field_id = field_data.get('id', index)
            desc = field_data.get('desc', '')
            action = field_data.get('action', '')
            
            # Enhanced question text extraction
            question_text = self._extract_question_text(field_data)
            
            # Enhanced type detection
            field_type = self._detect_field_type(field_data, desc, question_text)
            
            # Extract options for dropdown/radio/checkbox fields
            options = self._extract_field_options(field_data)
            
            # Determine if this is a required field
            required = field_data.get('required', True)
            
            field_info = {
                "index": field_id,
                "tag": self._get_field_tag(field_type),
                "type": field_type,
                "name": field_data.get('name', ''),
                "placeholder": field_data.get('placeholder', ''),
                "label": desc,
                "description": desc,
                "question_text": question_text,
                "action": action,
                "required": required,
                "options": options,
                "raw_html": str(field_data)
            }
            
            log_step(f"🔍 Enhanced field {field_id}: {question_text} (type: {field_type}, options: {len(options)})", symbol="🔍")
            return field_info
            
        except Exception as e:
            log_error(f"Failed to extract enhanced field info for index {index}", e)
            return None

    def _extract_question_text(self, field_data: Dict[str, Any]) -> str:
        """
        Extract the actual question text from field data, with fallback strategies.
        """
        try:
            # Strategy 1: Look for explicit question text fields
            for key in ['question_text', 'question', 'label', 'title', 'text', 'prompt']:
                if key in field_data and field_data[key]:
                    text = str(field_data[key]).strip()
                    if text and text.lower() not in ['text input', 'input', 'field', 'id']:
                        return text
            
            # Strategy 2: Look for aria-label or similar accessibility attributes
            for key in ['aria-label', 'ariaLabel', 'aria-describedby', 'ariaDescription']:
                if key in field_data and field_data[key]:
                    text = str(field_data[key]).strip()
                    if text and text.lower() not in ['text input', 'input', 'field', 'id']:
                        return text
            
            # Strategy 3: Check for parent/sibling text in the HTML structure
            if 'parent_text' in field_data and field_data['parent_text']:
                text = str(field_data['parent_text']).strip()
                if text and text.lower() not in ['text input', 'input', 'field', 'id']:
                    return text
            
            # Strategy 4: Look for surrounding text or context
            if 'context' in field_data and field_data['context']:
                text = str(field_data['context']).strip()
                if text and text.lower() not in ['text input', 'input', 'field', 'id']:
                    return text
            
            # Strategy 5: Use description if it's meaningful
            desc = field_data.get('desc', '')
            if desc and desc.lower() not in ['text input', 'input', 'field', 'id']:
                return desc.strip()
            
            # Strategy 6: Look for placeholder text
            placeholder = field_data.get('placeholder', '')
            if placeholder:
                return f"Field for: {placeholder}"
            
            # Strategy 7: Try to get more detailed information from the page
            # For Google Forms, we need to look deeper into the DOM structure
            raw_html = str(field_data)
            
            # Strategy 8: Try to extract from raw HTML using patterns
            import re
            
            # Look for common Google Forms patterns
            patterns = [
                r'aria-label[=:]\s*["\']([^"\']+)["\']',
                r'data-item-id[=:]\s*["\']([^"\']+)["\']',
                r'data-params[=:]\s*["\']([^"\']+)["\']',
                r'jsname[=:]\s*["\']([^"\']+)["\']',
                r'role[=:]\s*["\']([^"\']+)["\']',
                r'placeholder[=:]\s*["\']([^"\']+)["\']',
                r'title[=:]\s*["\']([^"\']+)["\']'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, raw_html, re.IGNORECASE)
                for match in matches:
                    if match and match.lower() not in ['text input', 'input', 'field', 'id', 'textbox']:
                        return match.strip()
            
            # Strategy 9: Try to infer from field index and context
            field_id = field_data.get('id', '')
            if field_id:
                # For Google Forms, try to get the actual question by analyzing the page structure
                # This would require additional DOM analysis
                return f"Form field {field_id}"
            
            # Strategy 10: Last resort - use a generic but descriptive name
            return "Form input field"
            
        except Exception as e:
            log_error(f"Failed to extract question text", e)
            return "Unknown field"

    def _detect_field_type(self, field_data: Dict[str, Any], desc: str, question_text: str) -> str:
        """
        Enhanced field type detection using multiple sources of information.
        """
        try:
            # Combine all text sources for better detection
            all_text = f"{desc} {question_text}".lower()
            
            # Check for specific field types in the data
            if 'type' in field_data:
                field_type = str(field_data['type']).lower()
                if field_type in ['select', 'dropdown', 'radio', 'checkbox', 'textarea', 'email', 'tel', 'date', 'number']:
                    return field_type
            
            # Check for options/choices (indicates dropdown/radio/checkbox)
            if 'options' in field_data and field_data['options']:
                if len(field_data['options']) == 1:
                    return 'radio'
                else:
                    return 'dropdown'
            
            # Check for action type
            action = field_data.get('action', '').lower()
            if 'select' in action or 'dropdown' in action:
                return 'dropdown'
            elif 'radio' in action or 'button' in action:
                return 'radio'
            elif 'checkbox' in action:
                return 'checkbox'
            
            # Text-based detection
            if 'email' in all_text:
                return 'email'
            elif 'phone' in all_text or 'mobile' in all_text or 'tel' in all_text:
                return 'tel'
            elif 'date' in all_text or 'birth' in all_text or 'dob' in all_text:
                return 'date'
            elif 'number' in all_text or 'amount' in all_text:
                return 'number'
            elif 'textarea' in all_text or 'message' in all_text or 'comment' in all_text:
                return 'textarea'
            elif 'dropdown' in all_text or 'select' in all_text:
                return 'dropdown'
            elif 'radio' in all_text:
                return 'radio'
            elif 'checkbox' in all_text:
                return 'checkbox'
            elif 'file' in all_text or 'upload' in all_text:
                return 'file'
            elif 'url' in all_text or 'website' in all_text:
                return 'url'
            else:
                # Add type mapping block to ensure field type is never empty
                desc_lower = desc.lower()
                if "text" in desc_lower:
                    field_type = "text"
                elif "radio" in desc_lower:
                    field_type = "radio"
                elif "check" in desc_lower:
                    field_type = "checkbox"
                elif "drop" in desc_lower:
                    field_type = "dropdown"
                else:
                    field_type = "text"  # default fallback
                
                return field_type
                
        except Exception as e:
            log_error("Failed to detect field type", e)
            return 'text'

    def _extract_field_options(self, field_data: Dict[str, Any]) -> List[str]:
        """
        Extract options from field data for dropdowns, radio buttons, and checkboxes.
        Enhanced to handle various data formats.
        """
        options = []
        
        try:
            # Check for options in various formats
            if 'options' in field_data:
                options = field_data['options']
            elif 'choices' in field_data:
                options = field_data['choices']
            elif 'values' in field_data:
                options = field_data['values']
            elif 'items' in field_data:
                options = field_data['items']
            
            # Ensure options are strings and filter out empty ones
            if options:
                options = [str(opt).strip() for opt in options if opt and str(opt).strip()]
            
            # If no options found, try to extract from raw HTML
            if not options:
                raw_html = str(field_data)
                import re
                # Look for option patterns in HTML
                option_patterns = [
                    r'<option[^>]*>([^<]+)</option>',
                    r'value[:\s]*["\']([^"\']+)["\']',
                    r'text[:\s]*["\']([^"\']+)["\']'
                ]
                
                for pattern in option_patterns:
                    matches = re.findall(pattern, raw_html, re.IGNORECASE)
                    for match in matches:
                        if match.strip() and match.strip().lower() not in ['select', 'option', 'choose']:
                            options.append(match.strip())
            
            return options
            
        except Exception as e:
            log_error("Failed to extract field options", e)
            return []

    def _get_field_tag(self, field_type: str) -> str:
        """
        Get HTML tag for field type.
        """
        tag_mapping = {
            'dropdown': 'select',
            'select': 'select',
            'radio': 'input',
            'checkbox': 'input',
            'textarea': 'textarea',
            'file': 'input',
            'email': 'input',
            'tel': 'input',
            'date': 'input',
            'number': 'input',
            'url': 'input',
            'text': 'input'
        }
        return tag_mapping.get(field_type, 'input')

    def _convert_elements_to_llm_format(self, elements_result: Any) -> List[Dict[str, Any]]:
        """
        Convert browser elements result to structured format for LLM processing.
        Enhanced to extract actual question text and detect field types properly.
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
                    
                    # Use enhanced field extraction
                    field_info = self._extract_enhanced_field_info(form, form_index)
                    if field_info:
                        form_fields.append(field_info)
                
                # Also check for individual input elements that might not be in forms
                inputs = elements_result.get('inputs', [])
                for input_index, input_field in enumerate(inputs):
                    field_info = self._extract_enhanced_field_info(input_field, len(form_fields) + input_index)
                    if field_info:
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
                            "question_text": line.strip(),
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
            return 'dropdown'
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
        Create robust prompt for LLM field mapping, using question_text, type, and options for each field.
        """
        prompt = f"""
    You are an expert form-filling assistant.

    You are given:
    1. A user instruction: "{instruction}"
    2. A list of form fields, each with:
    - index
    - question_text (visible field label or question)
    - type (text, dropdown, radio, checkbox, etc.)
    - options (if any, for selection fields)

    Goal:
    - Match each field to the appropriate value from the instruction using semantic understanding.
    - For fields labeled generically (e.g., "Form field 2"), look for more meaningful labels in `raw_html` or nearby text.
    - For dropdown/radio/checkbox, match value to one of the options (use fuzzy match if needed).
    - For text fields, extract value as-is from instruction.
    - Return a JSON array with fields: index, value, confidence (0–1), reason, question_text, field_type.

    Use examples like:
    - Question: "What is your name?" + Instruction: "name shubhangi mishra" → "shubhangi mishra"
    - Question: "Are you married?" + Instruction: "yes i am married" → "yes"
    - Question: "Which course are you taking?" + Instruction: "EAGv1 course" → "EAGv1"

    Do not return empty values. Only include mappings you are confident about.

    Output JSON:
    [
    {{
        "index": 0,
        "value": "shubhangi mishra",
        "confidence": 0.9,
        "reason": "Mapped from 'name' in instruction to 'What is your name?'",
        "question_text": "What is your name?",
        "field_type": "text"
    }},
    ...
    ]
    """

        return prompt

    def _parse_llm_field_mapping_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response to extract field mappings, including reason and confidence.
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
                        'confidence': mapping.get('confidence', 0.5),
                        'field_type': mapping.get('field_type', 'text'),
                        'reason': mapping.get('reason', ''),
                        'question_text': mapping.get('question_text', '')
                    })
            
            log_step(f"✅ Parsed {len(valid_mappings)} valid field mappings", symbol="✅")
            return valid_mappings
            
        except Exception as e:
            log_error("Failed to parse LLM field mapping response", e)
            return []

    async def _execute_field_filling(self, field_mapping: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the field filling instructions from LLM with enhanced field type handling and logging of mapping reasons.
        """
        log_step(f"📝 Executing field filling for {len(field_mapping)} fields", symbol="📝")
        
        fill_results = []
        
        for mapping in field_mapping:
            try:
                field_index = mapping['index']
                field_value = mapping['value']
                field_name = mapping.get('field_name', '')
                confidence = mapping.get('confidence', 0.5)
                field_type = mapping.get('field_type', 'text')
                reason = mapping.get('reason', '')
                question_text = mapping.get('question_text', 'unknown')
                
                # ✅ Add debug logs for better field matching trace
                log_step(f"🔗 Field match: '{question_text}' → '{field_value}' (confidence: {confidence})", symbol="🧠")
                
                log_step(f"📝 Filling field {field_index} ({field_name}) with '{field_value}' (type: {field_type}, confidence: {confidence}, reason: {reason})", symbol="📝")
                
                # Execute the fill action based on field type
                if self.multi_mcp:
                    result = await self._fill_field_by_type(field_index, field_value, field_type, field_name)
                    fill_results.append({
                        'field_index': field_index,
                        'field_name': field_name,
                        'value': field_value,
                        'confidence': confidence,
                        'field_type': field_type,
                        'reason': reason,
                        'question_text': question_text,
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
                        'field_type': field_type,
                        'reason': reason,
                        'question_text': question_text,
                        'status': 'success',
                        'result': {'message': f'Mock fill: {field_value} (type: {field_type})'}
                    })
                
            except Exception as e:
                log_error(f"Failed to fill field {field_index}", e)
                fill_results.append({
                    'field_index': field_index,
                    'field_name': field_name,
                    'value': field_value,
                    'confidence': confidence,
                    'field_type': field_type,
                    'reason': reason,
                    'question_text': question_text,
                    'status': 'error',
                    'result': str(e)
                })
        
        return fill_results

    async def _fill_field_by_type(self, field_index: int, field_value: str, field_type: str, field_name: str) -> Dict[str, Any]:
        """
        Fill a field based on its type with enhanced type detection and option handling.
        """
        try:
            log_step(f"📝 Filling field {field_index} ({field_type}) with '{field_value}' (confidence: {field_name})", symbol="📝")
            
            # Handle date field conversion
            if field_type == 'date':
                field_value = self._convert_date_format(field_value)
                log_step(f"📅 Converted date format: '{field_value}'", symbol="📅")
            
            # Click the field to focus first
            await self.multi_mcp.function_wrapper("click_element_by_index", field_index)
            
            # Handle different field types
            if field_type in ['dropdown', 'select']:
                return await self._fill_dropdown_field(field_index, field_value, field_name)
            elif field_type == 'radio':
                return await self._fill_radio_field(field_index, field_value, field_name)
            elif field_type == 'checkbox':
                return await self._fill_checkbox_field(field_index, field_value, field_name)
            elif field_type == 'textarea':
                return await self._fill_textarea_field(field_index, field_value, field_name)
            elif field_type in ['email', 'tel', 'date', 'number', 'url', 'text']:
                return await self._fill_text_field(field_index, field_value, field_name)
            else:
                # Default to text field
                log_step(f"⚠️ Unknown field type '{field_type}', treating as text", symbol="⚠️")
                return await self._fill_text_field(field_index, field_value, field_name)
                
        except Exception as e:
            log_error(f"Failed to fill field {field_index} by type", e)
            return {'error': str(e)}

    async def _fill_dropdown_field(self, field_index: int, field_value: str, field_name: str) -> Dict[str, Any]:
        """
        Fill a dropdown field by getting options and selecting the best match.
        """
        try:
            log_step(f"📋 Filling dropdown field {field_index} with '{field_value}'", symbol="📋")
            
            # Get dropdown options
            options_result = await self.multi_mcp.function_wrapper("get_dropdown_options", field_index)
            log_step(f"📋 Available options: {options_result}", symbol="📋")
            
            # Try to select the option
            select_result = await self.multi_mcp.function_wrapper("select_dropdown_option", field_index, field_value)
            
            return {
                'type': 'dropdown',
                'value': field_value,
                'options_result': options_result,
                'select_result': select_result
            }
            
        except Exception as e:
            log_error(f"Failed to fill dropdown field {field_index}", e)
            return {'error': str(e)}

    async def _fill_radio_field(self, field_index: int, field_value: str, field_name: str) -> Dict[str, Any]:
        """
        Fill a radio button field by clicking the appropriate option.
        """
        try:
            log_step(f"🔘 Filling radio field {field_index} with '{field_value}'", symbol="🔘")
            
            # For radio buttons, we need to find the correct option and click it
            # This might require additional logic to find the right radio button
            # For now, try clicking the field and then sending the value
            
            # Click the field to focus
            await self.multi_mcp.function_wrapper("click_element_by_index", field_index)
            
            # Try to send the value (this might work for some radio implementations)
            result = await self.multi_mcp.function_wrapper("input_text", field_index, field_value)
            
            return {
                'type': 'radio',
                'value': field_value,
                'result': result
            }
            
        except Exception as e:
            log_error(f"Failed to fill radio field {field_index}", e)
            return {'error': str(e)}

    async def _fill_checkbox_field(self, field_index: int, field_value: str, field_name: str) -> Dict[str, Any]:
        """
        Fill a checkbox field by checking/unchecking based on the value.
        """
        try:
            log_step(f"☑️ Filling checkbox field {field_index} with '{field_value}'", symbol="☑️")
            
            # For checkboxes, we need to determine if it should be checked
            should_check = self._should_check_checkbox(field_value)
            
            if should_check:
                # Click to check the checkbox
                result = await self.multi_mcp.function_wrapper("click_element_by_index", field_index)
            else:
                # Click to uncheck the checkbox (if it's already checked)
                result = await self.multi_mcp.function_wrapper("click_element_by_index", field_index)
            
            return {
                'type': 'checkbox',
                'value': field_value,
                'checked': should_check,
                'result': result
            }
            
        except Exception as e:
            log_error(f"Failed to fill checkbox field {field_index}", e)
            return {'error': str(e)}

    async def _fill_textarea_field(self, field_index: int, field_value: str, field_name: str) -> Dict[str, Any]:
        """
        Fill a textarea field with the provided value.
        """
        try:
            log_step(f"📝 Filling textarea field {field_index} with '{field_value}'", symbol="📝")
            
            # Click the field to focus first
            await self.multi_mcp.function_wrapper("click_element_by_index", field_index)
            await self.multi_mcp.function_wrapper("wait", 0.5)
            
            # Clear the field by selecting all and deleting
            await self.multi_mcp.function_wrapper("send_keys", "Control+a")
            await self.multi_mcp.function_wrapper("wait", 0.2)
            await self.multi_mcp.function_wrapper("send_keys", "Delete")
            await self.multi_mcp.function_wrapper("wait", 0.2)
            
            # Fill the textarea
            result = await self.multi_mcp.function_wrapper("input_text", field_index, field_value)
            
            return {
                'type': 'textarea',
                'value': field_value,
                'result': result
            }
            
        except Exception as e:
            log_error(f"Failed to fill textarea field {field_index}", e)
            return {'error': str(e)}

    async def _fill_text_field(self, field_index: int, field_value: str, field_name: str) -> Dict[str, Any]:
        """
        Fill a text field with the provided value.
        """
        try:
            log_step(f"📝 Filling text field {field_index} with '{field_value}'", symbol="📝")
            
            # Click the field to focus first
            await self.multi_mcp.function_wrapper("click_element_by_index", field_index)
            await self.multi_mcp.function_wrapper("wait", 0.5)
            
            # Clear the field by selecting all and deleting
            await self.multi_mcp.function_wrapper("send_keys", "Control+a")
            await self.multi_mcp.function_wrapper("wait", 0.2)
            await self.multi_mcp.function_wrapper("send_keys", "Delete")
            await self.multi_mcp.function_wrapper("wait", 0.2)
            
            # Fill the field
            result = await self.multi_mcp.function_wrapper("input_text", field_index, field_value)
            
            return {
                'type': 'text',
                'value': field_value,
                'result': result
            }
            
        except Exception as e:
            log_error(f"Failed to fill text field {field_index}", e)
            return {'error': str(e)}

    def _should_check_checkbox(self, field_value: str) -> bool:
        """
        Determine if a checkbox should be checked based on the field value.
        """
        if not field_value:
            return False
        
        # Convert to lowercase for comparison
        value_lower = str(field_value).lower().strip()
        
        # Positive values that should check the checkbox
        positive_values = ['yes', 'true', '1', 'on', 'checked', 'agree', 'accept', 'confirm']
        
        return value_lower in positive_values

    def _convert_date_format(self, user_input_str: str) -> str:
        """
        Convert various date formats to a standardized format using dateutil.parser.
        This makes date field handling more resilient.
        """
        try:
            if not user_input_str:
                return user_input_str
            
            # Try to parse the date using dateutil.parser
            parsed_date = parse(user_input_str)
            # Convert to DD/MM/YYYY format
            return parsed_date.strftime('%d/%m/%Y')
        except Exception as e:
            # Fallback if parse fails
            log_step(f"⚠️ Date parsing failed for '{user_input_str}', using original value: {e}", symbol="⚠️")
            return user_input_str

    def _is_form_filling_task(self, instruction: str, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Detect if the current task is a form filling task with enhanced detection.
        """
        instruction_lower = instruction.lower()
        
        # Check for form-related keywords in instruction
        form_keywords = [
            'fill form', 'submit form', 'complete form', 'fill out form',
            'fill in form', 'enter form', 'fill up form', 'fill the form',
            'submit', 'enter', 'fill', 'complete', 'fill out', 'fill in',
            'registration', 'sign up', 'signup', 'register', 'application',
            'survey', 'questionnaire', 'feedback', 'contact form'
        ]
        
        # Check for Google Forms URLs
        google_forms_indicators = [
            'forms.gle', 'docs.google.com/forms', 'google.com/forms',
            'forms.google.com', 'drive.google.com/forms'
        ]
        
        # Check for common form hosting platforms
        form_platforms = [
            'typeform.com', 'surveymonkey.com', 'jotform.com', 'wufoo.com',
            'formspree.io', 'netlify.com/forms', 'form.io', '123formbuilder.com'
        ]
        
        # Check instruction for form keywords
        has_form_keywords = any(keyword in instruction_lower for keyword in form_keywords)
        
        # Check for form-related tools
        form_tools = ['input_text', 'get_interactive_elements', 'click_element_by_index']
        is_form_tool = tool_name in form_tools
        
        # Check for form-related parameters
        form_parameters = ['text', 'index', 'option_text']
        has_form_parameters = any(param in parameters for param in form_parameters)
        
        # Check if we're on a form page (if we have browser state)
        is_form_page = self._detect_form_page()
        
        # Enhanced detection logic
        if has_form_keywords:
            log_step(f"✅ Form filling task detected by keywords: {[k for k in form_keywords if k in instruction_lower]}", symbol="🔍")
            return True
        
        if is_form_tool and has_form_parameters:
            log_step(f"✅ Form filling task detected by tool usage: {tool_name}", symbol="🔍")
            return True
        
        if is_form_page:
            log_step("✅ Form filling task detected by page analysis", symbol="🔍")
            return True
        
        # Check for Google Forms specifically
        if any(indicator in instruction_lower for indicator in google_forms_indicators):
            log_step("✅ Google Forms task detected", symbol="🔍")
            return True
        
        # Check for form platforms
        if any(platform in instruction_lower for platform in form_platforms):
            log_step(f"✅ Form platform task detected", symbol="🔍")
            return True
        
        return False

    def _detect_form_page(self) -> bool:
        """
        Detect if the current page is likely a form page.
        """
        try:
            # This would require access to the current page state
            # For now, we'll return False and let the other detection methods handle it
            # In a full implementation, this would analyze the current page content
            return False
        except Exception as e:
            log_error("Failed to detect form page", e)
            return False

    async def _enhance_form_detection(self) -> Dict[str, Any]:
        """
        Enhanced form detection with detailed analysis.
        """
        try:
            if not self.multi_mcp:
                return {"is_form": False, "reason": "No MCP connection"}
            
            # Get page structure for analysis
            page_structure = await self.multi_mcp.function_wrapper("get_enhanced_page_structure")
            
            if not isinstance(page_structure, dict):
                return {"is_form": False, "reason": "Invalid page structure"}
            
            # Analyze for form indicators
            form_indicators = {
                "has_forms": "forms" in page_structure and len(page_structure.get("forms", [])) > 0,
                "has_inputs": "inputs" in page_structure and len(page_structure.get("inputs", [])) > 0,
                "has_selects": "selects" in page_structure and len(page_structure.get("selects", [])) > 0,
                "has_radio_groups": "radio_groups" in page_structure and len(page_structure.get("radio_groups", [])) > 0,
                "has_checkbox_groups": "checkbox_groups" in page_structure and len(page_structure.get("checkbox_groups", [])) > 0
            }
            
            # Count total form elements
            total_form_elements = sum(len(page_structure.get(key, [])) for key in ["forms", "inputs", "selects", "radio_groups", "checkbox_groups"])
            
            # Determine if this is a form page
            is_form = any(form_indicators.values()) or total_form_elements > 0
            
            # Get current URL for additional analysis
            current_url = ""
            try:
                browser_session = await self.multi_mcp.function_wrapper("get_session_snapshot")
                # Extract URL from session snapshot if available
                if isinstance(browser_session, str) and "url" in browser_session.lower():
                    # This is a simplified extraction - in practice, you'd parse the session properly
                    current_url = "detected"
            except:
                pass
            
            return {
                "is_form": is_form,
                "form_indicators": form_indicators,
                "total_form_elements": total_form_elements,
                "current_url": current_url,
                "reason": "Form elements detected" if is_form else "No form elements found"
            }
            
        except Exception as e:
            log_error("Failed to enhance form detection", e)
            return {"is_form": False, "reason": f"Error: {str(e)}"}

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
        Extract form fields from markdown content using heuristics for question text.
        """
        form_fields = []
        try:
            lines = [line.strip() for line in markdown_content.split('\n') if line.strip()]
            # Heuristic: lines ending with '?' or ':' or that look like prompts
            question_lines = [
                line for line in lines
                if line.endswith('?') or line.endswith(':') or (len(line) < 120 and any(w in line.lower() for w in ['name', 'email', 'phone', 'address', 'date', 'question', 'option', 'choose', 'select']))
            ]
            # Remove duplicates while preserving order
            seen = set()
            question_lines = [x for x in question_lines if not (x in seen or seen.add(x))]
            # Assign each question line to a field in order
            for idx, q in enumerate(question_lines):
                field_info = {
                    "index": idx,
                    "tag": "input",
                    "type": self._infer_field_type(q),
                    "name": "",
                    "placeholder": "",
                    "label": q,
                    "description": q,
                    "question_text": q,
                    "action": "",
                    "required": True,
                    "options": []
                }
                form_fields.append(field_info)
            log_step(f"🔍 Extracted {len(form_fields)} fields from markdown (question heuristics)", symbol="📊")
            return form_fields
        except Exception as e:
            log_error("Failed to extract from markdown", e)
            return []

    def _enhance_question_text_with_page_content(self, field_info: Dict[str, Any], page_info: Dict[str, Any]) -> str:
        """
        Enhance question text using page content analysis, including markdown question heuristics.
        """
        try:
            current_question = field_info.get('question_text', '')
            # If we already have a meaningful question, return it
            if current_question and current_question.lower() not in ['id', 'text input', 'input', 'field', f"form field {field_info.get('index', 'unknown')}"]:
                return current_question
            # Try to extract from markdown content using question heuristics
            if 'markdown_content' in page_info:
                markdown = page_info['markdown_content']
                lines = [line.strip() for line in markdown.split('\n') if line.strip()]
                question_lines = [
                    line for line in lines
                    if line.endswith('?') or line.endswith(':') or (len(line) < 120 and any(w in line.lower() for w in ['name', 'email', 'phone', 'address', 'date', 'question', 'option', 'choose', 'select']))
                ]
                # Remove duplicates while preserving order
                seen = set()
                question_lines = [x for x in question_lines if not (x in seen or seen.add(x))]
                idx = field_info.get('index', 0)
                if idx < len(question_lines):
                    return question_lines[idx]
            # Fallback to previous logic (session snapshot, etc.)
            # Try to extract from session snapshot
            if 'session_snapshot' in page_info:
                snapshot = page_info['session_snapshot']
                if isinstance(snapshot, str):
                    import re
                    patterns = [
                        r'aria-label[=:]\s*["\']([^"\']+)["\']',
                        r'placeholder[=:]\s*["\']([^"\']+)["\']',
                        r'title[=:]\s*["\']([^"\']+)["\']',
                        r'data-params[=:]\s*["\']([^"\']+)["\']'
                    ]
                    for pattern in patterns:
                        matches = re.findall(pattern, snapshot, re.IGNORECASE)
                        for match in matches:
                            if match and match.lower() not in ['text input', 'input', 'field', 'id']:
                                return match.strip()
            # Fallback to current question or generic description
            if current_question and current_question != 'id':
                return current_question
            return f"Form field {field_info.get('index', 'unknown')}"
        except Exception as e:
            log_error("Failed to enhance question text", e)
            return field_info.get('question_text', 'Unknown field')

    async def _submit_form(self) -> Dict[str, Any]:
        """
        Submit the form after filling all fields. Tries to find a submit button by text if possible.
        """
        log_step("📤 Submitting form", symbol="📤")
        try:
            if self.multi_mcp:
                # Try to find a submit button by text (improve this logic as needed)
                # This assumes you have a tool to get all buttons or interactive elements
                try:
                    elements_result = await self.multi_mcp.function_wrapper("get_interactive_elements", True, True, True)
                    submit_index = None
                    if isinstance(elements_result, dict):
                        # Try to find a button with text 'Submit', 'Send', etc.
                        buttons = elements_result.get('buttons', [])
                        
                        # ✅ Only target real form submission buttons
                        for button in buttons:
                            label = button.get("text", "").lower()
                            
                            # ✅ Skip Google authentication buttons
                            if "sign in" in label.lower() or "sign into" in label.lower():
                                log_step(f"⚠️ Skipping Google auth button: {label}", symbol="⚠️")
                                continue
                            
                            # ✅ Only target real form submission buttons
                            if any(keyword in label for keyword in ["submit", "send", "next", "done", "submit form"]):
                                if not any(bad in label for bad in ["google", "drive", "sign in", "sign into"]):
                                    submit_index = button.get('id')
                                    log_step(f"📤 Found form submission button: {label}", symbol="✅")
                                    break
                        
                        # Fallback: try to find a button with 'submit' in text
                        if submit_index is None:
                            for btn in buttons:
                                btn_text = btn.get('text', '').strip().lower()
                                # Skip Google auth buttons in fallback too
                                if "sign in" in btn_text or "sign into" in btn_text:
                                    continue
                                if 'submit' in btn_text or 'send' in btn_text:
                                    submit_index = btn.get('id')
                                    log_step(f"📤 Found fallback submit button: {btn_text}", symbol="✅")
                                    break
                    
                    # Fallback: use index 0 if nothing found
                    if submit_index is None:
                        submit_index = 0
                        log_step("⚠️ No submit button found, using index 0", symbol="⚠️")
                    
                    log_step(f"📤 Clicking submit button at index {submit_index}", symbol="📤")
                    submit_result = await self.multi_mcp.function_wrapper("click_element_by_index", submit_index)
                    
                except Exception as e:
                    log_step(f"⚠️ Could not find submit button by text, defaulting to index 0: {e}", symbol="⚠️")
                    submit_result = await self.multi_mcp.function_wrapper("click_element_by_index", 0)
                
                log_step("✅ Form submitted successfully", symbol="✅")
                return {'status': 'success', 'result': submit_result}
            else:
                return {'status': 'success', 'result': {'message': 'Mock form submission'}}
        except Exception as e:
            log_error("Failed to submit form", e)
            return {'status': 'error', 'result': str(e)}

    async def _get_detailed_form_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed form information by analyzing the page content and DOM structure.
        This method tries to extract actual question text from Google Forms and other complex forms.
        """
        try:
            if not self.multi_mcp:
                return []
            
            # Get comprehensive page information
            page_info = {}
            
            # Try to get enhanced page structure
            try:
                enhanced_structure = await self.multi_mcp.function_wrapper("get_enhanced_page_structure")
                if isinstance(enhanced_structure, dict):
                    page_info['enhanced_structure'] = enhanced_structure
            except Exception as e:
                log_step(f"⚠️ Enhanced page structure failed: {e}", symbol="⚠️")
            
            # Try to get comprehensive markdown
            try:
                markdown_content = await self.multi_mcp.function_wrapper("get_comprehensive_markdown")
                if isinstance(markdown_content, str):
                    page_info['markdown_content'] = markdown_content
            except Exception as e:
                log_step(f"⚠️ Markdown content failed: {e}", symbol="⚠️")
            
            # Try to get session snapshot
            try:
                session_snapshot = await self.multi_mcp.function_wrapper("get_session_snapshot")
                if session_snapshot:
                    page_info['session_snapshot'] = session_snapshot
            except Exception as e:
                log_step(f"⚠️ Session snapshot failed: {e}", symbol="⚠️")
            
            # Get interactive elements
            try:
                interactive_elements = await self.multi_mcp.function_wrapper("get_interactive_elements", True, True, True)
                if isinstance(interactive_elements, dict):
                    page_info['interactive_elements'] = interactive_elements
            except Exception as e:
                log_step(f"⚠️ Interactive elements failed: {e}", symbol="⚠️")
            
            # Analyze the collected information to extract form details
            form_fields = []
            
            # Extract from interactive elements
            if 'interactive_elements' in page_info:
                elements = page_info['interactive_elements']
                forms = elements.get('forms', [])
                
                for i, form in enumerate(forms):
                    field_info = self._extract_enhanced_field_info(form, i)
                    if field_info:
                        # Try to enhance the question text using page content
                        enhanced_question = self._enhance_question_text_with_page_content(
                            field_info, page_info
                        )
                        field_info['question_text'] = enhanced_question
                        form_fields.append(field_info)
            
            # Extract from markdown content
            if 'markdown_content' in page_info and not form_fields:
                markdown_fields = self._extract_from_markdown(page_info['markdown_content'])
                form_fields.extend(markdown_fields)
            
            log_step(f"🔍 Detailed analysis extracted {len(form_fields)} fields", symbol="📊")
            return form_fields
            
        except Exception as e:
            log_error("Failed to get detailed form info", e)
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