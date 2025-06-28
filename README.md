# Browser Agent - Intelligent Web Automation System

A sophisticated multi-agent system for intelligent web automation, document processing, and task execution using AI-powered perception, decision-making, and browser control.

## 🌟 Overview

This project implements a comprehensive agentic AI system that can:
- **Perceive** user queries and route them to appropriate agents
- **Decide** on the best course of action for complex tasks
- **Execute** browser automation, form filling, and web scraping
- **Process** documents and extract information
- **Summarize** content and provide intelligent responses

The system uses a modular architecture with specialized agents for different types of tasks, powered by MCP (Model Context Protocol) servers and advanced AI models.

## 🏗️ Architecture

```
User Input → Perception → [BrowserAgent | Decision | Summarizer] → Action → Loop
```

### Core Components

1. **Perception System** - Analyzes and routes user queries
2. **BrowserAgent** - Handles web automation and form interactions
3. **Decision Agent** - Manages complex multi-step tasks
4. **Action Executor** - Executes specific actions and tools
5. **MCP Servers** - Provides various AI capabilities and tools

## 📁 Project Structure

### Core Agent Components

#### `/agent/` - Main Agent Loop and Session Management
- **`agent_loop3.py`** - Main agent execution loop with routing logic
- **`agentSession.py`** - Session management and state tracking
- **`contextManager.py`** - Context management for conversations
- **`model_manager.py`** - AI model configuration and management

#### `/agents/` - Specialized Agent Implementations
- **`browser_agent.py`** - Standalone browser automation agent
  - Multi-step web automation
  - Form detection and filling
  - Content extraction
  - Screenshot capture
  - Autonomous operation without Decision agent

#### `/perception/` - Query Analysis and Routing
- **`perception.py`** - Main perception system
  - Analyzes user queries
  - Routes to appropriate agents (Browser, Decision, Summarizer)
  - Priority-based routing logic
- **`perception_test.py`** - Unit tests for perception system

#### `/decision/` - Complex Task Planning
- **`decision.py`** - Decision-making agent for complex tasks
  - Multi-step planning
  - Tool selection and execution
  - Task decomposition
- **`decision_test.py`** - Decision agent tests

#### `/action/` - Action Execution System
- **`executor.py`** - Action execution engine
  - Tool execution management
  - State tracking
  - Error handling
- **`execute_step.py`** - Individual step execution
- **`sandbox_state/`** - Execution state snapshots

### Browser Automation System

#### `/browserMCP/` - Browser Control and MCP Integration
- **`browser_mcp_sse.py`** - Server-Sent Events MCP server
- **`browser_mcp_stdio.py`** - Standard I/O MCP server
- **`mcp_tools.py`** - Browser automation tools (676 lines)
- **`utils.py`** - Browser utilities and helpers

##### `/browserMCP/browser/` - Browser Session Management
- **`browser.py`** - Browser instance management
- **`session.py`** - Session handling
- **`profile.py`** - Browser profile configuration
- **`context.py`** - Browser context management
- **`extensions.py`** - Browser extension support

##### `/browserMCP/dom/` - DOM Processing
- **`service.py`** - DOM analysis services
- **`buildDomTree.js`** - DOM tree construction
- **`clickable_element_processor/`** - Interactive element detection
- **`history_tree_processor/`** - Navigation history processing

##### `/browserMCP/agent/` - Browser Agent Components
- **`views.py`** - Agent view handlers
- **`prompts.py`** - Agent-specific prompts
- **`memory/`** - Memory management for browser sessions

##### `/browserMCP/controller/` - Browser Control
- **`service.py`** - Control services
- **`registry/`** - Tool registry management

##### `/browserMCP/telemetry/` - Monitoring and Analytics
- **`service.py`** - Telemetry services
- **`views.py`** - Telemetry data views

### MCP Servers and Tools

#### `/mcp_servers/` - Multi-MCP Server Management
- **`multiMCP.py`** - Multi-MCP dispatcher and coordinator
- **`models.py`** - MCP model definitions
- **`mcp_server_1.py`** - Document processing server
- **`mcp_server_2.py`** - Web tools and search server
- **`mcp_server_3.py`** - Image processing server
- **`mcp_server_4.py`** - Additional tools server
- **`captioning_text.py`** - Text captioning service

##### `/mcp_servers/documents/` - Document Processing
- **`conversation_history/`** - Conversation memory storage
- **`faiss_index/`** - Vector search index
- **`images/`** - Document images
- Various document files (PDFs, DOCX, etc.)

##### `/mcp_servers/tools/` - Specialized Tools
- **`web_tools_async.py`** - Asynchronous web tools
- **`switch_search_method.py`** - Search method switching
- **`difficult_websites.txt`** - Problematic website list

### Supporting Systems

#### `/memory/` - Memory and Session Management
- **`memory_indexer.py`** - Memory indexing system
- **`memory_search.py`** - Memory search functionality
- **`session_logs/`** - Session logging and storage
- **`session_summaries_index/` - Session summary indexing

#### `/summarization/` - Content Summarization
- **`summarizer.py`** - Main summarization engine
- **`summarizer_test.py`** - Summarization tests

#### `/heuristics/` - Decision Heuristics
- **`heuristics.py`** - Heuristic-based decision making

#### `/utils/` - Utility Functions
- **`utils.py`** - General utility functions
- **`json_parser.py`** - JSON parsing utilities

#### `/prompts/` - AI Prompts
- **`browser_agent_prompt.txt`** - Browser agent prompts
- **`decision_prompt.txt`** - Decision agent prompts
- **`perception_prompt.txt`** - Perception system prompts
- **`summarizer_prompt.txt`** - Summarization prompts
- **`prompt_check.py`** - Prompt validation

#### `/config/` - Configuration Files
- **`mcp_server_config.yaml`** - MCP server configuration
- **`models.json`** - AI model configurations
- **`profiles.yaml`** - User profiles

### Testing and Documentation

#### `/tests/` - Test Suite
- **`browser_agent_test.py`** - Browser agent tests
- **`__init__.py`** - Test package initialization

#### `/media/` - Media Storage
- **`screenshots/`** - Browser screenshots
- **`pdf/`** - PDF documents

#### `/dom_dumps/` - DOM Snapshots
- Various DOM snapshot files for debugging

## 🚀 Key Features

### BrowserAgent Capabilities
- **Multi-step Automation**: Handles complex web workflows
- **Form Interaction**: Automatic form detection and filling
- **Content Extraction**: Extract data from web pages
- **Screenshot Capture**: Visual documentation
- **Navigation Control**: Tab management and URL navigation
- **Error Recovery**: Retry logic and error handling

### Perception System
- **Intelligent Routing**: Routes queries to appropriate agents
- **Priority-based Logic**: Browser → Summarize → Decision
- **Context Awareness**: Maintains conversation context
- **Keyword Detection**: Identifies browser automation tasks

### Decision Agent
- **Multi-step Planning**: Breaks complex tasks into steps
- **Tool Selection**: Chooses appropriate tools for tasks
- **State Management**: Tracks execution progress
- **Error Handling**: Manages failures and retries

### MCP Integration
- **Multiple Servers**: Coordinated MCP server management
- **Tool Registry**: Dynamic tool discovery and registration
- **Async Operations**: Non-blocking tool execution
- **State Persistence**: Maintains execution state

## 🧠 LLM-Powered Form Filling System

### Overview
The BrowserAgent includes an intelligent LLM-powered form filling system that can automatically extract form data from natural language instructions and match fields semantically. This solves the issue where form fields were being filled in order rather than being matched to the correct fields based on their labels or identifiers.

### Key Features

#### 🧠 Intelligent Form Data Extraction
- Automatically extracts form data from natural language instructions
- Supports various data formats (names, emails, dates, phone numbers, etc.)
- Handles complex instructions with multiple data points

#### 🔍 Semantic Field Matching
- Matches field names semantically (e.g., "email" matches "e-mail address")
- Recognizes field types (text, email, date, phone, etc.)
- Handles various form field naming conventions
- Supports fuzzy matching for typos and variations

#### 🎯 Automatic Form Detection
- Detects form filling tasks based on keywords and URLs
- Automatically triggers enhanced form interaction
- Supports Google Forms, contact forms, surveys, and more

### How It Works

#### 1. Form Task Detection
The system automatically detects when a task involves form filling by checking:
- **Keywords**: "fill", "form", "submit", "enter", "complete"
- **URLs**: Google Forms, survey sites, registration pages
- **Tools**: input_text, click_element_by_index, get_interactive_elements

#### 2. Data Extraction
Form data is extracted from natural language instructions using pattern matching:

```python
# Example instruction
"Fill form with name shubhangi mishra and email shubhimishra666@gmail.com and yes i am married and i am taking EAG course mainly EAGv1 course and my date of birth is 3rd october 1992"

# Extracted data
{
    "name": "shubhangi mishra",
    "email": "shubhimishra666@gmail.com", 
    "marital_status": "married",
    "course": "EAG course mainly EAGv1 course",
    "date_of_birth": "3rd october 1992"
}
```

#### 3. Field Matching
Fields are matched using multiple strategies:

**Semantic Matching**
- "email" matches "e-mail address", "mail", "email_address"
- "name" matches "full name", "first name", "last name", "given name"
- "date_of_birth" matches "birth date", "dob", "birthday", "born"

**Type Recognition**
- Email fields: type="email", name contains "email"
- Date fields: type="date", name contains "date", "birth", "dob"
- Phone fields: type="tel", name contains "phone", "mobile", "cell"

**Label Matching**
- Field labels, placeholders, and names are analyzed
- Fuzzy string matching handles variations and typos

#### 4. Intelligent Filling
Each field is filled with the correct value based on the best match found:

```python
# Example form elements
[0]<input type="text" name="full_name" placeholder="Enter your full name">
[1]<input type="email" name="email_address" placeholder="Enter your email">
[2]<input type="date" name="birth_date" placeholder="Date of birth">
[3]<select name="marital_status"><option>Single</option><option>Married</option></select>

# Field matching results
"name" -> index 0 (full_name field)
"email" -> index 1 (email_address field)  
"date_of_birth" -> index 2 (birth_date field)
"marital_status" -> index 3 (marital_status select)
```

### Supported Field Types

#### Personal Information
- **Name**: name, full name, first name, last name, given name, surname
- **Email**: email, e-mail, email address, e-mail address, mail
- **Phone**: phone, telephone, mobile, cell, phone number, mobile number
- **Date of Birth**: date of birth, birth date, dob, birthday, born
- **Age**: age, years old, current age
- **Gender**: gender, sex, male/female
- **Marital Status**: marital status, married, single, relationship status

#### Address Information
- **Address**: address, street address, home address, mailing address
- **City**: city, town, municipality, locality
- **State**: state, province, region, county
- **Zip Code**: zip code, postal code, zip, postcode, pincode
- **Country**: country, nation, nationality

#### Professional Information
- **Company**: company, organization, employer, workplace, business
- **Job Title**: job title, position, role, designation, title
- **Course**: course, program, study, education, training

#### Other Fields
- **Message**: message, comment, description, notes, feedback
- **Subject**: subject, topic, title, subject line
- **Website**: website, url, web address, site
- **Password**: password, pass, pwd, secret
- **Confirm Password**: confirm password, retype password, password confirmation

## 🔧 Browser Automation Tools

The BrowserAgent uses a comprehensive set of browser automation tools based on the browser-use library:

### Available Tools

```json
{"click_element_by_index": {"index": 44}}
{"input_text": {"index": 45, "text": "myusername"}}
{"go_back": {}}
{"scroll_down": {"amount": 500}}
{"scroll_up": {"amount": 500}}
{"open_tab": {"url": "https://example.com"}}
{"close_tab": {"page_id": 1}}
{"switch_tab": {"page_id": 1}}
{"save_pdf": {}}
{"get_dropdown_options": {"index": 12}}
{"select_dropdown_option": {"index": 12, "text": "Option Text"}}
{"extract_content": {"goal": "extract all emails", "should_strip_link_urls": true}}
{"drag_drop": {
    "element_source": "xpath_or_css_selector",
    "element_target": "xpath_or_css_selector",
    "element_source_offset": null,
    "element_target_offset": null,
    "coord_source_x": null,
    "coord_source_y": null,
    "coord_target_x": null,
    "coord_target_y": null,
    "steps": 10,
    "delay_ms": 5
}}
{"send_keys": {"keys": "Enter"}}
{"scroll_to_text": {"text": "Some visible text"}}
```

### Tool Descriptions

- **click_element_by_index**: Click a button, link, etc. by its index
- **input_text**: Type text into an input or textarea by its index
- **go_back**: Go back in browser history
- **scroll_down / scroll_up**: Scroll the page by a pixel amount
- **open_tab**: Open a new tab with a given URL
- **close_tab**: Close a tab by its page ID
- **switch_tab**: Switch to a tab by its page ID
- **save_pdf**: Save the current page as a PDF
- **get_dropdown_options**: Get all options from a dropdown by index
- **select_dropdown_option**: Select a dropdown option by index and visible text
- **extract_content**: Extract structured content from the page
- **drag_drop**: Drag and drop between elements or coordinates
- **send_keys**: Send special keys (e.g., Enter, Escape, Ctrl+T)
- **scroll_to_text**: Scroll to a specific visible text on the page

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.11+
- Ollama with required models
- Playwright for browser automation
- Required system dependencies

### Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd Browser_Agent
   uv sync  # Install dependencies
   ```

2. **Environment Configuration**:
   ```bash
   # Create .env file with required configurations
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start BrowserMCP Server**:
   ```bash
   uv run browserMCP/browser_mcp_sse.py
   ```

4. **Run Main Application**:
   ```bash
   uv run main.py
   ```

### Configuration Files

- **`config/mcp_server_config.yaml`**: MCP server configurations
- **`config/models.json`**: AI model settings
- **`config/profiles.yaml`**: User profile configurations

### Important Setup Notes

Check the following before running:
- Your `.env` file is present and properly configured
- All paths are correct in configuration files
- Required Ollama models are installed and initialized
- Run `uv run browserMCP/browser_mcp_sse.py` BEFORE running `uv run main.py`
- For debug mode: `uv run mcp dev browserMCP/browser_mcp_stdio.py`

## 📖 Usage Examples

### Browser Automation
```python
# Navigate to website and fill form
"Go to https://example.com/contact and fill out the contact form"

# Extract content
"Extract all product prices from the website"

# Take screenshot
"Visit https://example.com and take a screenshot"
```

### Form Filling Examples
```python
# Basic form filling
instruction = "Fill form https://forms.gle/example with name John Doe and email john@example.com"

# Complex form with multiple fields
instruction = """
Fill the registration form with:
- Name: Jane Smith
- Email: jane@example.com  
- Phone: 123-456-7890
- Date of birth: 15th March 1990
- Company: ABC Corp
- Course: Data Science
- Message: I'm interested in learning more
"""

# Form with marital status
instruction = "Complete the survey with name Mike Johnson, email mike@example.com, and yes I am married"
```

### Document Processing
```python
# Process PDF documents
"Summarize the content of the uploaded PDF"

# Search local documents
"Find information about Anmol Singh in local documents"
```

### Complex Tasks
```python
# Multi-step research
"Research the latest BMW 7 series vs 5 series differences and create a markdown report"
```

## 🧪 Testing

### Run All Tests
```bash
uv run pytest tests/
```

### Specific Test Categories
```bash
# Browser agent tests
uv run pytest tests/browser_agent_test.py

# Perception tests
uv run pytest perception/perception_test.py

# Decision tests
uv run pytest decision/decision_test.py
```

### Demo Scripts
```bash
# Browser agent demo
uv run demo_browser_agent.py

# Test browser session
uv run test_browser_session.py
```

## 🔧 Development

### Project Dependencies
Key dependencies include:
- `mcp[cli]>=1.7.0` - Model Context Protocol
- `playwright>=1.52.0` - Browser automation
- `llama-index>=0.12.31` - Document processing
- `faiss-cpu>=1.10.0` - Vector search
- `pydantic>=2.11.3` - Data validation
- `rich>=14.0.0` - Rich console output

### Development Workflow
1. **Feature Development**: Create feature branches
2. **Testing**: Write comprehensive tests
3. **Documentation**: Update relevant documentation
4. **Integration**: Test with full system
5. **Deployment**: Merge to main branch

## 📊 Performance and Monitoring

### Telemetry
- Session logging in `/memory/session_logs/`
- Performance metrics in `/browserMCP/telemetry/`
- State snapshots in `/action/sandbox_state/`

### Debugging
- DOM snapshots in `/dom_dumps/`
- Screenshots in `/media/screenshots/`
- Detailed logs with rich console output

## 🔄 Recent Updates

### BrowserAgent Redesign (Latest)
- Complete decoupling from Decision agent
- Standalone multi-step browser automation
- Enhanced form interaction capabilities
- Comprehensive test suite
- Improved error handling and retry logic

### Key Improvements
- **Autonomous Operation**: BrowserAgent operates independently
- **Multi-step Execution**: Internal loops for complex tasks
- **Form Intelligence**: Advanced form detection and filling
- **Clean Architecture**: Clear separation of concerns
- **Comprehensive Testing**: Full test coverage

### Form Filling Fixes
- **Function Wrapper Parameter Error**: Fixed parameter conversion issues
- **Insufficient Field Detection**: Enhanced form structure extraction
- **JSON Serialization Error**: Added custom JSON encoder for CallToolResult objects
- **Field Filling Improvements**: Added field focusing and wait times
- **Session Log JSON Parsing Error**: Fixed serialization across all components

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 🎯 Benefits Achieved

1. **Performance**: Browser tasks execute faster without Decision agent overhead
2. **Reliability**: Dedicated browser automation with specialized error handling
3. **Maintainability**: Clean separation between browser and non-browser tasks
4. **Scalability**: Independent BrowserAgent can be enhanced without affecting Decision agent
5. **Testing**: Comprehensive test coverage for all browser scenarios
6. **User Experience**: More efficient and reliable web automation

## 🚀 Running the System

### Demo Script
```bash
python demo_browser_agent.py
```

### Tests
```bash
pytest tests/browser_agent_test.py -v
```

### Integration
The BrowserAgent is automatically integrated into the main agent loop and will be used whenever Perception detects browser-related tasks.

## 📋 Summary

The BrowserAgent system redesign successfully achieves all objectives:

✅ **Standalone Operation**: BrowserAgent operates completely independently  
✅ **Multi-step Capability**: Handles complex, multi-turn browser tasks  
✅ **Form Interaction**: Advanced form detection, filling, and submission  
✅ **Clean Architecture**: No Decision agent involvement for browser tasks  
✅ **Comprehensive Testing**: Full test coverage with real-world scenarios  
✅ **Documentation**: Complete implementation and usage documentation  

The system now provides a robust, efficient, and maintainable solution for browser automation that is completely decoupled from the Decision agent while maintaining full integration with the existing agent architecture.

---

**Note**: This system is designed for research and development purposes. Ensure compliance with website terms of service when using browser automation features.


