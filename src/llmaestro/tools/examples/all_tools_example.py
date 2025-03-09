"""Example demonstrating how to use the all_tools module.

This example shows how to:
1. Access all available tools
2. Get tools by category
3. Use specific tools
4. Discover available categories
"""

import asyncio
from sqlalchemy import create_engine, text
from typing import Dict, Any, List

from llmaestro.tools.all_tools import (
    get_all_tools,
    get_tools_by_category,
    get_tool,
    get_categories,
    register_all_tools
)


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
    """Main function demonstrating the use of the all_tools module."""
    # Set up the database
    engine = await setup_database()
    
    # Re-register all tools with our database engine
    register_all_tools(sql_engine=engine)
    
    # Get all available tools
    all_tools = get_all_tools()
    print(f"Total tools available: {len(all_tools)}")
    
    # Get all categories
    categories = get_categories()
    print(f"Available categories: {', '.join(categories)}")
    
    # Get tools by category
    for category in categories:
        tools = get_tools_by_category(category)
        print(f"\nTools in category '{category}' ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
    
    # Get and use a specific tool
    read_only_sql = get_tool("execute_read_only_sql")
    if read_only_sql:
        print("\nExecuting SQL query:")
        try:
            results = await read_only_sql.source(
                query="SELECT * FROM users WHERE age > :min_age",
                params={"min_age": 25}
            )
            print(f"Query results:")
            for row in results:
                print(f"  {row['name']} ({row['email']}), Age: {row['age']}")
        except Exception as e:
            print(f"Error executing query: {e}")
    
    # Try to use a placeholder tool
    read_file = get_tool("read_file")
    if read_file:
        print("\nTrying to use a placeholder tool:")
        try:
            content = await read_file.source(path="example.txt")
            print(f"File content: {content}")
        except NotImplementedError as e:
            print(f"Expected error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 