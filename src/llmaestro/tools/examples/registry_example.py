"""Example demonstrating how to use the ToolRegistry with SQL tools.

This example shows how to:
1. Register SQL tools with the registry
2. Retrieve tools from the registry
3. Use the registry as a tool to discover available tools
"""

import asyncio
from sqlalchemy import create_engine, text
from typing import Dict, Any, List

from llmaestro.tools.registry import get_registry
from llmaestro.tools.sql_tools import create_sql_read_only_tool, create_sql_read_write_tool


async def setup_database():
    """Set up a sample SQLite database for the example."""
    # Create an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    
    # Create a sample table
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            age INTEGER
        )
        """))
        
        # Insert some sample data
        conn.execute(text("""
        INSERT INTO users (name, email, age) VALUES
        ('Alice', 'alice@example.com', 30),
        ('Bob', 'bob@example.com', 25),
        ('Charlie', 'charlie@example.com', 35)
        """))
        
        conn.commit()
    
    return engine


async def main():
    """Main function demonstrating the use of the ToolRegistry with SQL tools."""
    # Set up the database
    engine = await setup_database()
    
    # Get the registry instance
    registry = get_registry()
    
    # Create SQL tools
    read_only_tool = create_sql_read_only_tool(
        engine,
        name="query_database",
        description="Execute a read-only SQL query against the users database."
    )
    
    read_write_tool = create_sql_read_write_tool(
        engine,
        name="modify_database",
        description="Execute an SQL query that can modify the users database."
    )
    
    # Register the tools with the registry
    registry.register_tool("query_database", read_only_tool, category="database")
    registry.register_tool("modify_database", read_write_tool, category="database")
    
    # Register a custom function directly with the decorator
    @registry.register(name="count_users", category="database")
    async def count_users() -> int:
        """Count the number of users in the database."""
        tool = registry.get_tool("query_database")
        result = await tool.execute(query="SELECT COUNT(*) as count FROM users")
        return result[0]["count"]
    
    # Use the tools through the registry
    query_tool = registry.get_tool("query_database")
    modify_tool = registry.get_tool("modify_database")
    count_tool = registry.get_tool("count_users")
    
    # Execute a query
    print("Querying users:")
    users = await query_tool.execute(query="SELECT * FROM users")
    for user in users:
        print(f"  {user['name']} ({user['email']}), Age: {user['age']}")
    
    # Add a new user
    print("\nAdding a new user:")
    await modify_tool.execute(
        query="INSERT INTO users (name, email, age) VALUES (:name, :email, :age)",
        params={"name": "Dave", "email": "dave@example.com", "age": 40}
    )
    
    # Count users
    user_count = await count_tool.execute()
    print(f"\nTotal users: {user_count}")
    
    # Use the registry as a tool to discover available tools
    print("\nDiscovering available tools:")
    discovery_tool = registry.get_tool("list_available_tools")
    all_tools = await discovery_tool.execute()
    print(f"Total tools: {all_tools['total_tools']}")
    print(f"Categories: {', '.join(all_tools['categories'])}")
    
    # Get tools in the database category
    print("\nDatabase tools:")
    database_tools = await discovery_tool.execute(category="database")
    for tool in database_tools["tools"]:
        print(f"  {tool['name']}: {tool['description']}")


if __name__ == "__main__":
    asyncio.run(main()) 