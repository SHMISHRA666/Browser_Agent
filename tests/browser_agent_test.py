import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from agents.browser_agent import BrowserAgent, build_browser_agent_input
from agent.agentSession import AgentSession


class TestBrowserAgent:
    """Test suite for BrowserAgent functionality - completely decoupled from Decision agent."""
    
    @pytest.fixture
    def mock_multi_mcp(self):
        """Create a mock MultiMCP instance."""
        mock_mcp = Mock()
        mock_mcp.function_wrapper = AsyncMock()
        
        # Create mock tools with proper name attributes and schemas
        mock_tools = []
        tool_names = ["open_tab", "go_to_url", "click_element_by_index", 
                     "input_text", "get_interactive_elements", 
                     "get_enhanced_page_structure", "take_screenshot", "done"]
        
        for name in tool_names:
            mock_tool = Mock()
            mock_tool.name = name
            # Add schema for parameter extraction
            if name in ["open_tab", "go_to_url"]:
                mock_tool.inputSchema = {
                    "properties": {
                        "url": {"type": "string"}
                    }
                }
            elif name == "click_element_by_index":
                mock_tool.inputSchema = {
                    "properties": {
                        "index": {"type": "integer"}
                    }
                }
            elif name == "input_text":
                mock_tool.inputSchema = {
                    "properties": {
                        "index": {"type": "integer"},
                        "text": {"type": "string"}
                    }
                }
            elif name == "get_interactive_elements":
                mock_tool.inputSchema = {
                    "properties": {
                        "viewport_mode": {"type": "string"},
                        "strict_mode": {"type": "boolean"}
                    }
                }
            else:
                mock_tool.inputSchema = {"properties": {}}
            
            mock_tools.append(mock_tool)
        
        mock_mcp.get_all_tools.return_value = mock_tools
        
        # Add tool_map for parameter extraction
        mock_mcp.tool_map = {}
        for tool in mock_tools:
            mock_mcp.tool_map[tool.name] = {"tool": tool}
        
        return mock_mcp
    
    @pytest.fixture
    def browser_agent(self, mock_multi_mcp):
        """Create a BrowserAgent instance with mocked dependencies."""
        return BrowserAgent(multi_mcp=mock_multi_mcp)
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock AgentSession."""
        return Mock(spec=AgentSession)
    
    def test_browser_agent_initialization(self, mock_multi_mcp):
        """Test BrowserAgent initialization."""
        agent = BrowserAgent(multi_mcp=mock_multi_mcp)
        
        assert agent.multi_mcp == mock_multi_mcp
        assert agent.max_steps == 15
        assert agent.max_retries == 3
        assert agent.browser_prompt_path == "prompts/browser_agent_prompt.txt"
    
    @pytest.mark.asyncio
    async def test_build_step_input(self, browser_agent):
        """Test building step input for BrowserAgent."""
        execution_state = {
            "instruction": "Go to example.com and fill form",
            "current_step": 2,
            "completed_steps": [{"step": 1}, {"step": 2}],
            "failed_steps": [],
            "browser_state": {"current_url": "https://example.com"}
        }
        
        step_input = browser_agent._build_step_input(execution_state, 2)
        
        assert step_input["step_number"] == 3
        assert step_input["max_steps"] == 15
        assert step_input["original_instruction"] == "Go to example.com and fill form"
        assert len(step_input["completed_steps"]) == 2
        assert step_input["browser_state"]["current_url"] == "https://example.com"
        assert "available_tools" in step_input
    
    @pytest.mark.asyncio
    async def test_get_available_tools(self, browser_agent):
        """Test getting available tools from MultiMCP."""
        tools = browser_agent._get_available_tools()
        
        expected_tools = ["open_tab", "go_to_url", "click_element_by_index", 
                         "input_text", "get_interactive_elements", 
                         "get_enhanced_page_structure", "take_screenshot", "done"]
        assert tools == expected_tools
    
    def test_extract_url_from_result(self, browser_agent):
        """Test extracting URL from tool results."""
        # Test with URL in result
        result_with_url = {"url": "https://example.com", "status": "success"}
        url = browser_agent._extract_url_from_result(result_with_url)
        assert url == "https://example.com"
        
        # Test with current_url
        result_with_current_url = {"current_url": "https://test.com"}
        url = browser_agent._extract_url_from_result(result_with_current_url)
        assert url == "https://test.com"
        
        # Test with no URL
        result_no_url = {"status": "success"}
        url = browser_agent._extract_url_from_result(result_no_url)
        assert url == ""
    
    def test_serialize_result(self, browser_agent):
        """Test result serialization for JSON compatibility."""
        # Test basic types
        assert browser_agent._serialize_result("hello") == "hello"
        assert browser_agent._serialize_result(42) == 42
        assert browser_agent._serialize_result(True) is True
        assert browser_agent._serialize_result(None) is None
        
        # Test dict
        test_dict = {"key": "value", "number": 123}
        assert browser_agent._serialize_result(test_dict) == test_dict
        
        # Test list
        test_list = ["item1", "item2", 456]
        assert browser_agent._serialize_result(test_list) == test_list
        
        # Test CallToolResult-like object with JSON content
        class MockCallToolResult:
            def __init__(self, content_text):
                self.content = [MockContent(content_text)]
        
        class MockContent:
            def __init__(self, text):
                self.text = text
        
        mock_result = MockCallToolResult('{"url": "https://example.com", "status": "success"}')
        serialized = browser_agent._serialize_result(mock_result)
        assert isinstance(serialized, dict)
        assert serialized["url"] == "https://example.com"
        assert serialized["status"] == "success"
        
        # Test CallToolResult-like object with non-JSON content
        mock_result_text = MockCallToolResult("Success")
        serialized_text = browser_agent._serialize_result(mock_result_text)
        assert serialized_text == "Success"
        
        # Test complex nested structure
        nested = {
            "result": MockCallToolResult('{"data": {"items": [1, 2, 3]}}'),
            "metadata": {"timestamp": "2023-01-01"}
        }
        serialized_nested = browser_agent._serialize_result(nested)
        assert isinstance(serialized_nested, dict)
        assert "result" in serialized_nested
        assert "metadata" in serialized_nested
        assert serialized_nested["metadata"]["timestamp"] == "2023-01-01"
    
    def test_check_task_completion(self, browser_agent):
        """Test task completion detection."""
        execution_state = {"completed_steps": [{"step": 1}, {"step": 2}]}
        
        # Test with task_complete in step plan
        step_plan = {"task_complete": True}
        result = {"status": "success"}
        is_complete = browser_agent._check_task_completion(step_plan, result, execution_state)
        assert is_complete is True
        
        # Test with done tool
        step_plan = {"tool_name": "done"}
        is_complete = browser_agent._check_task_completion(step_plan, result, execution_state)
        assert is_complete is True
        
        # Test with success result
        step_plan = {"task_complete": False}
        result = {"success": True}
        is_complete = browser_agent._check_task_completion(step_plan, result, execution_state)
        assert is_complete is True
        
        # Test with insufficient steps
        step_plan = {"task_complete": False}
        result = {"status": "success"}
        execution_state = {"completed_steps": [{"step": 1}]}
        is_complete = browser_agent._check_task_completion(step_plan, result, execution_state)
        assert is_complete is False
    
    @pytest.mark.asyncio
    async def test_browser_agent_run_success(self, browser_agent, mock_session):
        """Test successful BrowserAgent execution."""
        # Mock the step plan generation
        mock_step_plan = {
            "action_type": "navigation",
            "tool_name": "open_tab",
            "parameters": {"url": "https://example.com"},
            "reasoning": "Navigate to the website",
            "expected_outcome": "Page loads",
            "task_complete": True
        }
        
        with patch.object(browser_agent, '_get_step_plan', return_value=mock_step_plan):
            with patch.object(browser_agent, '_execute_step_plan') as mock_execute:
                mock_execute.return_value = {
                    "status": "success",
                    "result": "Page loaded successfully",
                    "task_complete": True
                }
                
                result = await browser_agent.run("Go to example.com", session=mock_session)
                
                assert result["status"] == "completed"
                assert result["steps_completed"] == 1
                assert result["steps_failed"] == 0
                assert result["total_steps"] == 1
    
    @pytest.mark.asyncio
    async def test_browser_agent_run_failure(self, browser_agent, mock_session):
        """Test BrowserAgent execution with failures."""
        # Mock step plan generation failure
        with patch.object(browser_agent, '_get_step_plan', return_value=None):
            result = await browser_agent.run("Go to example.com", session=mock_session)
            
            assert result["status"] == "failed"
            assert "BrowserAgent failed due to repeated errors" in result["result"]
    
    @pytest.mark.asyncio
    async def test_browser_agent_run_timeout(self, browser_agent, mock_session):
        """Test BrowserAgent execution timeout."""
        # Mock step plan that never completes
        mock_step_plan = {
            "action_type": "navigation",
            "tool_name": "open_tab",
            "parameters": {"url": "https://example.com"},
            "reasoning": "Navigate to the website",
            "expected_outcome": "Page loads",
            "task_complete": False
        }
        
        with patch.object(browser_agent, '_get_step_plan', return_value=mock_step_plan):
            with patch.object(browser_agent, '_execute_step_plan') as mock_execute:
                mock_execute.return_value = {
                    "status": "success",
                    "result": "Step completed",
                    "task_complete": False
                }
                
                # Set max_steps to 1 to force timeout
                browser_agent.max_steps = 1
                result = await browser_agent.run("Go to example.com", session=mock_session)
                
                assert result["status"] == "timeout"
                assert result["total_steps"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_step_plan_success(self, browser_agent, mock_session):
        """Test successful step plan execution."""
        step_plan = {
            "tool_name": "open_tab",
            "parameters": {"url": "https://example.com"},
            "reasoning": "Navigate to website"
        }
        execution_state = {
            "browser_state": {},
            "current_step": 0,
            "completed_steps": [],
            "failed_steps": []
        }
        
        # Mock successful tool execution
        browser_agent.multi_mcp.function_wrapper.return_value = {
            "url": "https://example.com",
            "status": "success"
        }

        result = await browser_agent._execute_step_plan(step_plan, execution_state, mock_session)
        
        assert result["status"] == "success"
        assert result["result"]["url"] == "https://example.com"
        assert result["task_complete"] is False
    
    def test_find_form_field_index(self, browser_agent):
        """Test finding form field index."""
        elements_result = "[0]<input name='name'>\n[1]<input name='email'>\n[2]<textarea name='message'>"
        
        # Test finding name field
        index = browser_agent._find_form_field_index(elements_result, "name")
        assert index == 0
        
        # Test finding email field
        index = browser_agent._find_form_field_index(elements_result, "email")
        assert index == 1
        
        # Test finding non-existent field
        index = browser_agent._find_form_field_index(elements_result, "phone")
        assert index is None
    
    def test_find_submit_button_index(self, browser_agent):
        """Test finding submit button index."""
        elements_result = "[0]<input name='name'>\n[1]<button type='submit'>\n[2]<input type='button'>"
        
        index = browser_agent._find_submit_button_index(elements_result)
        assert index == 1
        
        # Test with no submit button
        elements_result = "[0]<input name='name'>\n[1]<input type='button'>"
        index = browser_agent._find_submit_button_index(elements_result)
        assert index is None
        
        # Test with button without type='submit'
        elements_result = "[0]<input name='name'>\n[1]<button>Submit</button>"
        index = browser_agent._find_submit_button_index(elements_result)
        assert index is None

    @pytest.mark.asyncio
    async def test_execute_step_plan_failure(self, browser_agent, mock_session):
        """Test step plan execution failure."""
        step_plan = {
            "tool_name": "invalid_tool",
            "parameters": {}
        }
        execution_state = {
            "browser_state": {},
            "current_step": 0,
            "completed_steps": [],
            "failed_steps": []
        }
        
        # Mock tool execution failure
        browser_agent.multi_mcp.function_wrapper.side_effect = Exception("Tool not found")
        
        result = await browser_agent._execute_step_plan(step_plan, execution_state, mock_session)
        
        assert result["status"] == "error"
        assert "Tool invalid_tool failed" in result["result"]

    @pytest.mark.asyncio
    async def test_handle_form_interaction(self, browser_agent, mock_session):
        """Test form interaction functionality."""
        form_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "message": "Test message"
        }
        execution_state = {
            "browser_state": {},
            "completed_steps": [],
            "failed_steps": []
        }
        
        # Mock get_interactive_elements
        browser_agent.multi_mcp.function_wrapper.side_effect = [
            "[0]<input name='name'>\n[1]<input name='email'>\n[2]<textarea name='message'>\n[3]<button type='submit'>",
            "Success",  # name input
            "Success",  # email input  
            "Success",  # message input
            "Success"   # submit button
        ]
        
        result = await browser_agent._handle_form_interaction(form_data, execution_state)
        
        assert result["status"] == "success"
        assert result["form_completed"] is True
        assert len(result["result"]) == 5  # 1 get_elements + 3 inputs + 1 submit


class TestBrowserAgentIntegration:
    """Integration tests for BrowserAgent with real scenarios."""
    
    @pytest.fixture
    def mock_multi_mcp(self):
        """Create a mock MultiMCP instance for integration tests."""
        mock_mcp = Mock()
        mock_mcp.function_wrapper = AsyncMock()
        
        # Create mock tools with proper name attributes and schemas
        mock_tools = []
        tool_names = ["open_tab", "go_to_url", "click_element_by_index", 
                     "input_text", "get_interactive_elements", 
                     "get_enhanced_page_structure", "take_screenshot", "done"]
        
        for name in tool_names:
            mock_tool = Mock()
            mock_tool.name = name
            # Add schema for parameter extraction
            if name in ["open_tab", "go_to_url"]:
                mock_tool.inputSchema = {
                    "properties": {
                        "url": {"type": "string"}
                    }
                }
            elif name == "click_element_by_index":
                mock_tool.inputSchema = {
                    "properties": {
                        "index": {"type": "integer"}
                    }
                }
            elif name == "input_text":
                mock_tool.inputSchema = {
                    "properties": {
                        "index": {"type": "integer"},
                        "text": {"type": "string"}
                    }
                }
            elif name == "get_interactive_elements":
                mock_tool.inputSchema = {
                    "properties": {
                        "viewport_mode": {"type": "string"},
                        "strict_mode": {"type": "boolean"}
                    }
                }
            else:
                mock_tool.inputSchema = {"properties": {}}
            
            mock_tools.append(mock_tool)
        
        mock_mcp.get_all_tools.return_value = mock_tools
        
        # Add tool_map for parameter extraction
        mock_mcp.tool_map = {}
        for tool in mock_tools:
            mock_mcp.tool_map[tool.name] = {"tool": tool}
        
        return mock_mcp
    
    @pytest.mark.asyncio
    async def test_build_browser_agent_input(self):
        """Test building browser agent input."""
        instruction = "Go to example.com and fill the contact form"
        browser_state = {"current_url": "https://example.com"}
        
        input_data = build_browser_agent_input(instruction, browser_state)
        
        assert input_data["instruction"] == instruction
        assert input_data["browser_state"] == browser_state
        assert "timestamp" in input_data
    
    @pytest.mark.asyncio
    async def test_browser_agent_without_multi_mcp(self):
        """Test BrowserAgent behavior without MultiMCP."""
        agent = BrowserAgent(multi_mcp=None)
        
        # Should handle gracefully
        tools = agent._get_available_tools()
        assert tools == []
        
        # Should raise error when trying to execute
        with pytest.raises(RuntimeError, match="MultiMCP not available"):
            await agent._execute_step_plan(
                {"tool_name": "test", "parameters": {}},
                {"browser_state": {}, "current_step": 0, "completed_steps": [], "failed_steps": []},
                Mock()
            )
    
    @pytest.mark.asyncio
    async def test_form_filling_scenario(self, mock_multi_mcp):
        """Test complete form filling scenario."""
        agent = BrowserAgent(multi_mcp=mock_multi_mcp)
        
        # Mock the entire form filling process
        mock_multi_mcp.function_wrapper.side_effect = [
            # Step 1: Open tab
            {"url": "https://example.com/contact", "status": "success"},
            # Step 2: Get interactive elements
            "[0]<input name='name'>\n[1]<input name='email'>\n[2]<textarea name='message'>\n[3]<button type='submit'>",
            # Step 3: Fill name
            "Success",
            # Step 4: Fill email
            "Success", 
            # Step 5: Fill message
            "Success",
            # Step 6: Submit form
            "Success"
        ]
        
        # Mock step plan generation
        with patch.object(agent, '_get_step_plan') as mock_plan:
            mock_plan.return_value = {
                "action_type": "navigation",
                "tool_name": "open_tab",
                "parameters": {"url": "https://example.com/contact"},
                "reasoning": "Navigate to contact page",
                "expected_outcome": "Contact page loads",
                "task_complete": False
            }
            
            result = await agent.run("Go to example.com/contact and fill the contact form")
            
            assert result["status"] == "completed"
            assert result["steps_completed"] >= 1
    
    @pytest.mark.asyncio
    async def test_multi_step_navigation(self, mock_multi_mcp):
        """Test multi-step navigation scenario."""
        agent = BrowserAgent(multi_mcp=mock_multi_mcp)
        
        # Mock multi-step navigation with proper side_effect
        mock_multi_mcp.function_wrapper.side_effect = [
            # Step 1: Open tab
            {"url": "https://google.com", "status": "success"},
            # Step 2: Search
            {"search_results": ["result1", "result2"], "status": "success"},     
            # Step 3: Click first result
            {"url": "https://example.com", "status": "success"},
            # Step 4: Get page content
            {"content": "Page content here", "status": "success"}
        ]
        
        # Mock step plan generation to return different plans for each step
        step_plans = [
            {
                "action_type": "navigation",
                "tool_name": "open_tab",
                "parameters": {"url": "https://google.com"},
                "reasoning": "Start navigation",
                "expected_outcome": "Google page loads",
                "task_complete": False
            },
            {
                "action_type": "interaction",
                "tool_name": "click_element_by_index",
                "parameters": {"index": 0},
                "reasoning": "Click first result",
                "expected_outcome": "Navigate to result page",
                "task_complete": True
            }
        ]
        
        with patch.object(agent, '_get_step_plan') as mock_plan:
            mock_plan.side_effect = step_plans
            
            result = await agent.run("Search for 'Python tutorials' on Google and click the first result")
            
            assert result["status"] == "completed"
            assert result["steps_completed"] >= 1


class TestBrowserAgentDecoupling:
    """Test that BrowserAgent is completely decoupled from Decision agent."""
    
    @pytest.fixture
    def mock_multi_mcp(self):
        """Create a mock MultiMCP instance for decoupling tests."""
        mock_mcp = Mock()
        mock_mcp.function_wrapper = AsyncMock()
        
        # Create mock tools with proper name attributes and schemas
        mock_tools = []
        tool_names = ["open_tab", "go_to_url", "click_element_by_index", 
                     "input_text", "get_interactive_elements", 
                     "get_enhanced_page_structure", "take_screenshot", "done"]
        
        for name in tool_names:
            mock_tool = Mock()
            mock_tool.name = name
            # Add schema for parameter extraction
            if name in ["open_tab", "go_to_url"]:
                mock_tool.inputSchema = {
                    "properties": {
                        "url": {"type": "string"}
                    }
                }
            elif name == "click_element_by_index":
                mock_tool.inputSchema = {
                    "properties": {
                        "index": {"type": "integer"}
                    }
                }
            elif name == "input_text":
                mock_tool.inputSchema = {
                    "properties": {
                        "index": {"type": "integer"},
                        "text": {"type": "string"}
                    }
                }
            elif name == "get_interactive_elements":
                mock_tool.inputSchema = {
                    "properties": {
                        "viewport_mode": {"type": "string"},
                        "strict_mode": {"type": "boolean"}
                    }
                }
            else:
                mock_tool.inputSchema = {"properties": {}}
            
            mock_tools.append(mock_tool)
        
        mock_mcp.get_all_tools.return_value = mock_tools
        
        # Add tool_map for parameter extraction
        mock_mcp.tool_map = {}
        for tool in mock_tools:
            mock_mcp.tool_map[tool.name] = {"tool": tool}
        
        return mock_mcp
    
    def test_no_decision_imports(self):
        """Test that BrowserAgent doesn't import Decision-related modules."""
        import agents.browser_agent
        
        # Check that no decision-related imports exist
        assert not hasattr(agents.browser_agent, 'Decision')
        assert not hasattr(agents.browser_agent, 'build_decision_input')
        
        # Check that BrowserAgent is self-contained
        assert hasattr(agents.browser_agent, 'BrowserAgent')
        assert hasattr(agents.browser_agent, 'build_browser_agent_input')
    
    def test_standalone_functionality(self, mock_multi_mcp):
        """Test that BrowserAgent works completely standalone."""
        agent = BrowserAgent(multi_mcp=mock_multi_mcp)
        
        # Should have all necessary methods for standalone operation
        assert hasattr(agent, 'run')
        assert hasattr(agent, '_execute_step')
        assert hasattr(agent, '_get_step_plan')
        assert hasattr(agent, '_execute_step_plan')
        assert hasattr(agent, '_handle_form_interaction')
        
        # Should not have any decision-related methods
        assert not hasattr(agent, 'decision')
        assert not hasattr(agent, '_run_decision_loop')
    
    @pytest.mark.asyncio
    async def test_independent_execution(self, mock_multi_mcp):
        """Test that BrowserAgent executes independently."""
        agent = BrowserAgent(multi_mcp=mock_multi_mcp)

        # Mock successful execution
        with patch.object(agent, '_get_step_plan') as mock_plan:
            with patch.object(agent, '_execute_step_plan') as mock_execute:      
                mock_plan.return_value = {
                    "action_type": "navigation",
                    "tool_name": "open_tab",
                    "parameters": {"url": "https://example.com"},
                    "reasoning": "Navigate to website",
                    "expected_outcome": "Page loads",
                    "task_complete": True
                }
                mock_execute.return_value = {
                    "status": "success",
                    "result": "Success",
                    "task_complete": True
                }

                # Should execute without any decision agent involvement
                result = await agent.run("Go to example.com")

                assert result["status"] == "completed"
                # Verify no decision-related calls were made - check the mock was called correctly
                assert mock_plan.call_count >= 1
                assert mock_execute.call_count >= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 