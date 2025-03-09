#!/usr/bin/env python
"""Example demonstrating the use of SQL tools in LLMaestro.

This example shows how to:
1. Create a SQLite database with a sample table
2. Create read-only and read-write SQL tools
3. Execute queries using these tools

Usage:
    poetry run python examples/sql_tool_example.py
"""

import os
import json
import asyncio
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field

from llmaestro.tools.sql_tools import create_sql_read_only_tool, create_sql_read_write_tool


def setup_database(db_path: str = "example.db"):
    """Set up a sample SQLite database with a users table."""
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create a new database
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Create a users table
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
    """Run the SQL tools example."""
    print("Setting up sample SQLite database...")
    engine = setup_database()
    
    # Create read-only and read-write SQL tools
    read_only_tool = create_sql_read_only_tool(engine)
    read_write_tool = create_sql_read_write_tool(engine)
    
    print("\n1. Using read-only SQL tool to query data:")
    try:
        # Execute a SELECT query
        results = await read_only_tool.execute(
            query="SELECT * FROM users WHERE age > :min_age",
            params={"min_age": 25}
        )
        print(f"Query results: {json.dumps(results, indent=2)}")
        
        # Try to execute an INSERT query (should fail)
        print("\nAttempting to execute INSERT with read-only tool (should fail):")
        await read_only_tool.execute(
            query="INSERT INTO users (name, email, age) VALUES ('Dave', 'dave@example.com', 40)"
        )
    except Exception as e:
        print(f"Error (expected): {e}")
    
    print("\n2. Using read-write SQL tool:")
    try:
        # Execute an INSERT query
        result = await read_write_tool.execute(
            query="INSERT INTO users (name, email, age) VALUES ('Dave', 'dave@example.com', 40)"
        )
        print(f"Insert result (rows affected): {result}")
        
        # Execute a SELECT query to verify the insert
        results = await read_write_tool.execute(
            query="SELECT * FROM users"
        )
        print(f"All users after insert: {json.dumps(results, indent=2)}")
        
        # Try to execute a dangerous query (should fail)
        print("\nAttempting to execute dangerous query with read-write tool (should fail):")
        await read_write_tool.execute(
            query="DROP TABLE users"
        )
    except Exception as e:
        print(f"Error (expected): {e}")
    
    # Clean up
    os.remove("example.db")
    print("\nExample completed and database cleaned up.")


if __name__ == "__main__":
    asyncio.run(main()) 