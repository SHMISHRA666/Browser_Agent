import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_browser_session():
    """Test basic browser session initialization"""
    print("🧪 Testing browser session initialization...")
    
    try:
        from browserMCP.mcp_utils.utils import ensure_browser_session, get_browser_session
        
        # Initialize browser session
        print("📡 Initializing browser session...")
        await ensure_browser_session()
        
        print("📡 Getting browser session...")
        browser_session = await get_browser_session()
        
        # Test basic operations
        print("📡 Getting current page...")
        page = await browser_session.get_current_page()
        print(f"✅ Browser session initialized successfully")
        print(f"   Current page URL: {page.url}")
        
        # Test navigation
        print("📡 Testing navigation...")
        await page.goto("https://www.google.com")
        print(f"   Navigated to: {page.url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Browser session test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_browser_session()) 