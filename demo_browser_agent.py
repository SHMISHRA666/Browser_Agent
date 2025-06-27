#!/usr/bin/env python3
"""
BrowserAgent Demo Script
Demonstrates the standalone BrowserAgent functionality without Decision agent involvement.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.browser_agent import BrowserAgent, build_browser_agent_input
from agent.agentSession import AgentSession
from utils.utils import log_step, log_error


async def test_browser_session():
    """Test basic browser session initialization"""
    print("🧪 Testing browser session initialization...")
    
    try:
        from browserMCP.mcp_utils.utils import ensure_browser_session, get_browser_session
        
        # Initialize browser session
        await ensure_browser_session()
        browser_session = await get_browser_session()
        
        # Test basic operations
        page = await browser_session.get_current_page()
        print(f"✅ Browser session initialized successfully")
        print(f"   Current page URL: {page.url}")
        
        # Test navigation
        await page.goto("https://www.google.com")
        print(f"   Navigated to: {page.url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Browser session test failed: {e}")
        return False

async def main():
    """Main demo function"""
    print("🌐 BrowserAgent Demo")
    print("=" * 50)
    
    # Test browser session first
    browser_ok = await test_browser_session()
    if not browser_ok:
        print("❌ Browser session test failed. Exiting.")
        return
    
    print("\n✅ Browser session test passed. Starting demo...\n")
    
    # Initialize BrowserAgent
    browser_agent = BrowserAgent()
    
    # Test cases
    test_cases = [
        {
            "name": "Basic Navigation",
            "instruction": "Open www.google.com and search for 'Python programming'",
            "expected_actions": ["open_tab", "go_to_url", "input_text", "click_element_by_index"]
        },
        {
            "name": "Form Interaction",
            "instruction": "Go to https://httpbin.org/forms/post and fill out the form with name 'John Doe' and age '30'",
            "expected_actions": ["go_to_url", "input_text", "click_element_by_index"]
        },
        {
            "name": "Multi-step Task",
            "instruction": "Navigate to https://example.com, take a screenshot, and get the page structure",
            "expected_actions": ["go_to_url", "take_screenshot", "get_enhanced_page_structure"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print("-" * 40)
        print(f"📝 Instruction: {test_case['instruction']}")
        
        try:
            # Run the agent
            result = await browser_agent.run(test_case['instruction'])
            
            # Display results
            print(f"\n📊 Results:")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Steps completed: {len(result.get('completed_steps', []))}")
            print(f"   Steps failed: {len(result.get('failed_steps', []))}")
            
            if result.get('completed_steps'):
                print(f"   Last action: {result['completed_steps'][-1].get('tool_name', 'unknown')}")
            
            if result.get('failed_steps'):
                print(f"   Errors: {len(result['failed_steps'])}")
                for error in result['failed_steps'][:2]:  # Show first 2 errors
                    print(f"     - {error.get('result', {}).get('error', 'Unknown error')}")
            
            print(f"✅ Test {i} completed")
            
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait between tests
        await asyncio.sleep(2)
    
    print(f"\n🎉 Demo completed!")
    print("=" * 50)


async def demo_browser_agent():
    """Demonstrate BrowserAgent functionality."""
    print("🌐 BrowserAgent Demo - Standalone Mode")
    print("=" * 50)
    
    # Initialize real MultiMCP with proper server configuration
    import yaml
    from mcp_servers.multiMCP import MultiMCP
    
    # Load MCP server configs
    with open("config/mcp_server_config.yaml", "r") as f:
        profile = yaml.safe_load(f)
        mcp_servers_list = profile.get("mcp_servers", [])
        configs = list(mcp_servers_list)
    
    # Initialize MultiMCP dispatcher
    multi_mcp = MultiMCP(server_configs=configs)
    await multi_mcp.initialize()
    
    # Create BrowserAgent instance with real MultiMCP
    browser_agent = BrowserAgent(multi_mcp=multi_mcp)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Simple Navigation",
            "instruction": "Go to https://example.com",
            "description": "Basic URL navigation"
        },
        {
            "name": "Form Filling",
            "instruction": "Go to https://example.com/contact and fill out the contact form",
            "description": "Multi-step form interaction"
        },
        {
            "name": "Content Extraction",
            "instruction": "Visit https://example.com and extract the main content",
            "description": "Content extraction from webpage"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Test Scenario {i}: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Instruction: {scenario['instruction']}")
        print("-" * 50)
        
        try:
            # Create session for this test
            session = AgentSession(f"demo_session_{i}", scenario['instruction'])
            
            # Run BrowserAgent
            result = await browser_agent.run(scenario['instruction'], session=session)
            
            # Display results
            print(f"✅ Status: {result['status']}")
            print(f"📊 Steps Completed: {result['steps_completed']}")
            print(f"❌ Steps Failed: {result['steps_failed']}")
            print(f"🔄 Total Steps: {result['total_steps']}")
            print(f"📝 Result: {result['result']}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 50)
    
    print("\n🎉 BrowserAgent Demo Completed!")
    print("The BrowserAgent operated completely independently of the Decision agent.")
    
    # Cleanup
    try:
        await multi_mcp.shutdown()
    except:
        pass


async def demo_perception_routing():
    """Demonstrate how Perception routes to BrowserAgent instead of Decision."""
    print("\n🧠 Perception Routing Demo")
    print("=" * 50)
    
    # Example queries that should route to BrowserAgent
    browser_queries = [
        "Go to https://example.com and fill out the contact form",
        "Visit Google and search for 'Python tutorials'",
        "Navigate to Amazon and find the price of iPhone",
        "Open GitHub and create a new repository",
        "Take a screenshot of the homepage",
        "Login to the website and download my profile"
    ]
    
    # Example queries that should route to Decision (non-browser)
    decision_queries = [
        "Calculate the sum of 1 to 100",
        "Process the local CSV file",
        "Analyze the text document",
        "Generate a report from local data"
    ]
    
    print("🌐 Queries that should route to BrowserAgent:")
    for i, query in enumerate(browser_queries, 1):
        print(f"   {i}. {query}")
    
    print("\n🤖 Queries that should route to Decision:")
    for i, query in enumerate(decision_queries, 1):
        print(f"   {i}. {query}")
    
    print("\n✅ The Perception module will automatically detect browser-related tasks")
    print("   and route them to BrowserAgent, bypassing the Decision agent entirely.")


def main():
    """Main demo function."""
    print("🚀 BrowserAgent System Redesign Demo")
    print("=" * 60)
    print("This demo shows the standalone BrowserAgent functionality")
    print("that is completely decoupled from the Decision agent.")
    print("=" * 60)
    
    # Run demos
    asyncio.run(demo_browser_agent())
    demo_perception_routing()
    
    print("\n📋 Summary of Changes Made:")
    print("✅ Created standalone BrowserAgent in agents/browser_agent.py")
    print("✅ Updated perception_prompt.txt to prioritize browser routing")
    print("✅ Modified perception.py to handle browser route")
    print("✅ Updated agent_loop3.py to bypass Decision for browser tasks")
    print("✅ Added comprehensive tests in tests/browser_agent_test.py")
    print("✅ BrowserAgent operates independently with multi-step capabilities")
    print("✅ Form interaction functionality included")
    print("✅ No references to 'Decision' for browser-related tasks")


if __name__ == "__main__":
    # Run the main demo function
    asyncio.run(demo_browser_agent()) 