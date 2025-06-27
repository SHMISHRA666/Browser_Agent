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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

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

---

**Note**: This system is designed for research and development purposes. Ensure compliance with website terms of service when using browser automation features.


