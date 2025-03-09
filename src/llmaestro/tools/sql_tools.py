"""SQL tools for LLMaestro.

This module provides tools for executing SQL queries against databases with appropriate
safety guards. It includes:
- SQLQueryParams: Pydantic model for SQL query parameters
- SQLReadOnlyGuard: Function guard that only allows read-only SQL queries
- SQLReadWriteGuard: Function guard that allows both read and write SQL queries
- create_sql_read_only_tool: Factory function to create a read-only SQL query tool
- create_sql_read_write_tool: Factory function to create a read-write SQL query tool
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable, cast
import re
import sqlalchemy
from pydantic import BaseModel, Field, validator
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import SQLAlchemyError

from llmaestro.tools.core import FunctionGuard, BasicFunctionGuard, ToolParams


class SQLQueryParams(BaseModel):
    """Parameters for executing an SQL query."""
    
    query: str = Field(
        description="The SQL query to execute. For read-only queries, use SELECT statements."
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional parameters to bind to the query for parameterized queries."
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate that the query is not empty."""
        if not v or not v.strip():
            raise ValueError("SQL query cannot be empty")
        return v


class SQLReadOnlyGuard(BasicFunctionGuard):
    """Function guard that only allows read-only SQL queries.
    
    This guard checks that SQL queries are read-only (SELECT statements)
    before executing them against the database.
    """
    
    def __init__(self, engine_or_connection: Union[Engine, Connection, Callable[[], Union[Engine, Connection]]]):
        """Initialize the SQL read-only guard.
        
        Args:
            engine_or_connection: SQLAlchemy engine, connection, or a callable that returns either
        """
        super().__init__(self._execute_query)
        self._engine_or_connection = engine_or_connection
        
    def is_safe_to_run(self, **kwargs) -> bool:
        """Check if the SQL query is safe to run (read-only).
        
        Args:
            **kwargs: Must include 'query' parameter
            
        Returns:
            bool: True if the query is read-only, False otherwise
        """
        query = kwargs.get('query', '')
        if not query:
            return False
            
        # Remove comments and normalize whitespace
        clean_query = self._clean_query(query)
        
        # Check if the query is read-only
        return self._is_read_only_query(clean_query)
    
    def _clean_query(self, query: str) -> str:
        """Remove comments and normalize whitespace in SQL query."""
        # Remove SQL comments
        query = re.sub(r'--.*?$', ' ', query, flags=re.MULTILINE)  # Single line comments
        query = re.sub(r'/\*.*?\*/', ' ', query, flags=re.DOTALL)  # Multi-line comments
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        return query.strip().lower()
    
    def _is_read_only_query(self, query: str) -> bool:
        """Check if a query is read-only (SELECT statement).
        
        Args:
            query: Cleaned and normalized SQL query
            
        Returns:
            bool: True if the query is read-only, False otherwise
        """
        # Check if query starts with SELECT
        if not query.startswith('select'):
            return False
            
        # Check for data modification keywords
        modification_keywords = [
            'insert', 'update', 'delete', 'drop', 'alter', 'create', 
            'truncate', 'replace', 'upsert', 'merge', 'grant', 'revoke'
        ]
        
        # Look for these keywords as standalone words
        for keyword in modification_keywords:
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, query):
                return False
                
        return True
    
    def _get_connection(self) -> Connection:
        """Get a SQLAlchemy connection from the engine or connection.
        
        Returns:
            Connection: SQLAlchemy connection
        """
        if callable(self._engine_or_connection):
            conn_or_engine = self._engine_or_connection()
        else:
            conn_or_engine = self._engine_or_connection
            
        # If it's an engine, get a connection
        if isinstance(conn_or_engine, Engine):
            return conn_or_engine.connect()
        else:
            return cast(Connection, conn_or_engine)
    
    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a read-only SQL query.
        
        Args:
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            List of dictionaries representing the query results
            
        Raises:
            ValueError: If the query is not read-only
            SQLAlchemyError: If there's an error executing the query
        """
        # Get connection
        conn = self._get_connection()
        conn_is_from_engine = isinstance(self._engine_or_connection, Engine)
        
        try:
            # Execute query
            result = conn.execute(sqlalchemy.text(query), params or {})
            
            # Convert result to list of dictionaries
            columns = result.keys()
            return_value = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Close connection if we created it from an engine
            if conn_is_from_engine:
                conn.close()
                
            return return_value
            
        except SQLAlchemyError as e:
            if conn_is_from_engine:
                conn.close()
            raise ValueError(f"Error executing SQL query: {str(e)}")


class SQLReadWriteGuard(SQLReadOnlyGuard):
    """Function guard that allows both read and write SQL queries.
    
    This guard allows all types of SQL queries but implements basic safety checks
    to prevent the most dangerous operations.
    """
    
    def __init__(self, engine_or_connection: Union[Engine, Connection, Callable[[], Union[Engine, Connection]]]):
        """Initialize the SQL read-write guard.
        
        Args:
            engine_or_connection: SQLAlchemy engine, connection, or a callable that returns either
        """
        # Initialize with the read-write execute query function
        super().__init__(engine_or_connection)
        # Override the function with our read-write version
        self._func = self._execute_read_write_query
    
    def is_safe_to_run(self, **kwargs) -> bool:
        """Check if the SQL query is safe to run.
        
        Args:
            **kwargs: Must include 'query' parameter
            
        Returns:
            bool: True if the query passes basic safety checks, False otherwise
        """
        query = kwargs.get('query', '')
        if not query:
            return False
            
        # Remove comments and normalize whitespace
        clean_query = self._clean_query(query)
        
        # Check for dangerous operations
        return not self._has_dangerous_operations(clean_query)
    
    def _has_dangerous_operations(self, query: str) -> bool:
        """Check if a query contains dangerous operations.
        
        Args:
            query: Cleaned and normalized SQL query
            
        Returns:
            bool: True if the query contains dangerous operations, False otherwise
        """
        # Check for system table access or dangerous operations
        dangerous_patterns = [
            # System tables/views (database specific, this is a general example)
            r'\binformation_schema\b', r'\bpg_\w+\b', r'\bsys\.\w+\b', r'\bsqlite_\w+\b',
            
            # Dangerous operations
            r'\bdrop\s+database\b', r'\btruncate\s+database\b',
            r'\bsystem\b', r'\bexec\b', r'\bshell\b',
            
            # Potential SQL injection patterns
            r';\s*select', r';\s*insert', r';\s*update', r';\s*delete',
            
            # File operations
            r'\bload\s+data\b', r'\binto\s+outfile\b', r'\binto\s+dumpfile\b'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query):
                return True
                
        return False
    
    def _execute_read_write_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Union[List[Dict[str, Any]], int]:
        """Execute an SQL query (read or write).
        
        Args:
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            For SELECT queries: List of dictionaries representing the query results
            For other queries: Number of rows affected
            
        Raises:
            ValueError: If there's an error executing the query
            SQLAlchemyError: If there's an error executing the query
        """
        # Get connection
        conn = self._get_connection()
        conn_is_from_engine = isinstance(self._engine_or_connection, Engine)
        
        # Check if it's a transaction that needs to be committed
        needs_commit = not self._is_read_only_query(self._clean_query(query))
        
        # Execute query
        try:
            result = conn.execute(sqlalchemy.text(query), params or {})
            
            if needs_commit:
                conn.commit()
                
            # For SELECT queries, return results as list of dictionaries
            if query.strip().lower().startswith('select'):
                columns = result.keys()
                return_value = [dict(zip(columns, row)) for row in result.fetchall()]
            else:
                # For other queries, return number of rows affected
                return_value = result.rowcount
                
            # Close connection if we created it from an engine
            if conn_is_from_engine:
                conn.close()
                
            return return_value
                
        except SQLAlchemyError as e:
            if needs_commit:
                conn.rollback()
            if conn_is_from_engine:
                conn.close()
            raise ValueError(f"Error executing SQL query: {str(e)}")


def create_sql_read_only_tool(
    engine_or_connection: Union[Engine, Connection, Callable[[], Union[Engine, Connection]]],
    name: str = "execute_read_only_sql",
    description: str = "Execute a read-only SQL query (SELECT only) against the database."
) -> ToolParams:
    """Create a tool for executing read-only SQL queries.
    
    Args:
        engine_or_connection: SQLAlchemy engine, connection, or callable that returns either
        name: Name of the tool
        description: Description of the tool
        
    Returns:
        ToolParams: Tool parameters for executing read-only SQL queries
    """
    guard = SQLReadOnlyGuard(engine_or_connection)
    
    return ToolParams(
        name=name,
        description=description,
        parameters=SQLQueryParams.model_json_schema(),
        return_type=List[Dict[str, Any]],
        source=guard
    )


def create_sql_read_write_tool(
    engine_or_connection: Union[Engine, Connection, Callable[[], Union[Engine, Connection]]],
    name: str = "execute_sql",
    description: str = "Execute an SQL query against the database. Can perform both read and write operations."
) -> ToolParams:
    """Create a tool for executing SQL queries (both read and write).
    
    Args:
        engine_or_connection: SQLAlchemy engine, connection, or callable that returns either
        name: Name of the tool
        description: Description of the tool
        
    Returns:
        ToolParams: Tool parameters for executing SQL queries
    """
    guard = SQLReadWriteGuard(engine_or_connection)
    
    return ToolParams(
        name=name,
        description=description,
        parameters=SQLQueryParams.model_json_schema(),
        return_type=Union[List[Dict[str, Any]], int],
        source=guard
    )
