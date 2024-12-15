import os
from typing import Dict, List, Tuple, Optional
import sqlite3
import psycopg2
import mysql.connector
from urllib.parse import urlparse
import streamlit as st
import requests
import json
import time

class DatabaseConnector:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
        self.db_type = self._detect_db_type()
    
    def _detect_db_type(self) -> str:
        """Detect database type from connection string."""
        if self.connection_string.startswith('sqlite'):
            return 'sqlite'
        elif self.connection_string.startswith('postgresql'):
            return 'postgresql'
        elif self.connection_string.startswith('mysql'):
            return 'mysql'
        else:
            raise ValueError("Unsupported database type")

    def connect(self):
        """Establish database connection based on type."""
        try:
            if self.db_type == 'sqlite':
                self.connection = sqlite3.connect(self.connection_string.replace('sqlite:///', ''))
            elif self.db_type == 'postgresql':
                parsed = urlparse(self.connection_string)
                self.connection = psycopg2.connect(
                    database=parsed.path[1:],
                    user=parsed.username,
                    password=parsed.password,
                    host=parsed.hostname,
                    port=parsed.port
                )
            elif self.db_type == 'mysql':
                parsed = urlparse(self.connection_string)
                self.connection = mysql.connector.connect(
                    database=parsed.path[1:],
                    user=parsed.username,
                    password=parsed.password,
                    host=parsed.hostname,
                    port=parsed.port
                )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {str(e)}")

    def extract_schema(self) -> Dict[str, List[Dict]]:
        """Extract database schema information."""
        schema = {}
        cursor = self.connection.cursor()
        
        if self.db_type == 'sqlite':
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                schema[table_name] = [
                    {
                        'name': col[1],
                        'type': col[2],
                        'nullable': not col[3],
                        'primary_key': bool(col[5])
                    }
                    for col in columns
                ]
                
        elif self.db_type == 'postgresql':
            # Get all tables in public schema
            cursor.execute("""
                SELECT table_name, column_name, data_type, is_nullable, 
                       (SELECT true FROM information_schema.key_column_usage k 
                        WHERE k.table_name=c.table_name AND k.column_name=c.column_name)
                FROM information_schema.columns c
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
            """)
            results = cursor.fetchall()
            
            current_table = None
            for row in results:
                table_name, column_name, data_type, is_nullable, is_primary = row
                if table_name != current_table:
                    current_table = table_name
                    schema[table_name] = []
                
                schema[table_name].append({
                    'name': column_name,
                    'type': data_type,
                    'nullable': is_nullable == 'YES',
                    'primary_key': bool(is_primary)
                })
                
        elif self.db_type == 'mysql':
            # Get all tables
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"DESCRIBE {table_name};")
                columns = cursor.fetchall()
                schema[table_name] = [
                    {
                        'name': col[0],
                        'type': col[1],
                        'nullable': col[2] == 'YES',
                        'primary_key': col[3] == 'PRI'
                    }
                    for col in columns
                ]
        
        cursor.close()
        return schema

    def execute_query(self, query: str) -> List[Dict]:
        """Execute SQL query and return results."""
        cursor = self.connection.cursor()
        cursor.execute(query)
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Fetch results and convert to list of dictionaries
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(column_names, row)))
        
        cursor.close()
        return results

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


class QueryProcessor:
    def __init__(self, schema: Dict[str, List[Dict]], ollama_url: str = "http://localhost:11434"):
        self.schema = schema
        self.ollama_url = ollama_url
        self.max_retries = 2

    def format_schema_for_prompt(self) -> str:
        """Format schema information for AI prompt."""
        schema_text = "Database Schema:\n\n"
        for table_name, columns in self.schema.items():
            schema_text += f"Table: {table_name}\n"
            for column in columns:
                pk_indicator = " (Primary Key)" if column['primary_key'] else ""
                nullable_indicator = " (Nullable)" if column['nullable'] else ""
                schema_text += f"- {column['name']}: {column['type']}{pk_indicator}{nullable_indicator}\n"
            schema_text += "\n"
        return schema_text

    def check_query_relevance(self, user_query: str) -> Tuple[bool, str]:
        """Check if the user query is relevant to the database schema."""
        prompt = f"""Given the following database schema and user query, determine if the query can be answered using this database structure.
        
{self.format_schema_for_prompt()}

User Query: {user_query}

Analyze if this query can be answered using the available tables and columns.
Respond with either:
RELEVANT: [explanation] if the query can be answered
NOT_RELEVANT: [explanation] if the query cannot be answered"""

        try:
            response = self._call_ollama(prompt)
            is_relevant = response.startswith("RELEVANT:")
            explanation = response.split(":", 1)[1].strip()
            return is_relevant, explanation
        except requests.exceptions.HTTPError as http_err:
            raise Exception(f"HTTP error occurred: {str(http_err)}")
        except Exception as e:
            raise Exception(f"Failed to check query relevance: {str(e)}")

    def generate_sql_query(self, natural_language_query: str) -> str:
        """Generate SQL query from natural language using Ollama."""
        prompt = f"""Given the following database schema and natural language query, generate a valid SQL query.
        
{self.format_schema_for_prompt()}

Natural Language Query: {natural_language_query}

Generate a SQL query that will answer this question. Consider:
1. Use appropriate JOINs when needed
2. Include WHERE clauses for filtering
3. Use aggregations (COUNT, SUM, etc.) when appropriate
4. Order results if relevant

Respond with only the SQL query, no explanations."""

        try:
            sql_query = self._call_ollama(prompt)
            return sql_query.strip()
        except Exception as e:
            raise Exception(f"Failed to generate SQL query: {str(e)}")

    def generate_human_response(self, query_results: List[Dict], user_query: str) -> str:
        """Generate a human-readable response from the query results."""
        prompt = f"""Given the following query results and the original question, generate a natural language response.

Original Question: {user_query}

Query Results: {json.dumps(query_results, indent=2)}

Provide a clear, concise answer in natural language that directly addresses the user's question."""

        try:
            return self._call_ollama(prompt)
        except Exception as e:
            raise Exception(f"Failed to generate human response: {str(e)}")

    def handle_error(self, error_message: str, original_query: str) -> str:
        """Generate a new SQL query based on the error message."""
        prompt = f"""Given the following error message and original query, generate a corrected SQL query.

Original Query: {original_query}
Error Message: {error_message}

Database Schema:
{self.format_schema_for_prompt()}

Generate a corrected SQL query that addresses the error. Respond with only the SQL query."""

        try:
            return self._call_ollama(prompt).strip()
        except Exception as e:
            raise Exception(f"Failed to generate corrected query: {str(e)}")

    def _call_ollama(self, prompt: str) -> str:
        """Make an API call to Ollama."""
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "deepseek-coder",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()

def init_session_state():
    """Initialize session state variables."""
    if 'db' not in st.session_state:
        st.session_state.db = None
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'schema' not in st.session_state:
        st.session_state.schema = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def main():
    st.title("AI Database Query Assistant")
    
    init_session_state()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    connection_string = st.sidebar.text_input(
        "Database Connection String",
        value="sqlite:///example.db",
        help="Format: sqlite:///path/to/db or postgresql://user:pass@host:port/db"
    )
    
    ollama_url = st.sidebar.text_input(
        "Ollama API URL",
        value="http://localhost:11434",
        help="URL where Ollama is running"
    )

    # Connect to database button
    if st.sidebar.button("Connect to Database"):
        try:
            with st.spinner("Connecting to database..."):
                st.session_state.db = DatabaseConnector(connection_string)
                st.session_state.db.connect()
                st.session_state.schema = st.session_state.db.extract_schema()
                st.session_state.processor = QueryProcessor(st.session_state.schema, ollama_url)
                st.sidebar.success("Connected successfully!")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {str(e)}")

    # Main area
    if st.session_state.db and st.session_state.processor:
        # Display schema
        with st.expander("Database Schema", expanded=False):
            st.text(st.session_state.processor.format_schema_for_prompt())
        
        # Query input
        st.subheader("Ask Your Question")
        user_query = st.text_area("Type your question in natural language", 
                                help="Ask anything about your data, and I'll try to find the answer!")
        
        if st.button("Get Answer"):
            if user_query:
                try:
                    # Check query relevance
                    with st.spinner("Checking query relevance..."):
                        is_relevant, explanation = st.session_state.processor.check_query_relevance(user_query)
                        
                    if not is_relevant:
                        st.warning(f"I can't answer this question using the available data: {explanation}")
                        return

                    # Generate and execute query with retry logic
                    for attempt in range(st.session_state.processor.max_retries):
                        try:
                            with st.spinner("Generating SQL query..."):
                                sql_query = st.session_state.processor.generate_sql_query(user_query)
                                st.code(sql_query, language="sql")

                            with st.spinner("Executing query..."):
                                results = st.session_state.db.execute_query(sql_query)

                            with st.spinner("Generating response..."):
                                human_response = st.session_state.processor.generate_human_response(results, user_query)
                                
                            # Display results
                            st.success(human_response)
                            with st.expander("View Raw Results"):
                                st.json(results)
                            
                            # Store in history
                            st.session_state.query_history.append({
                                "question": user_query,
                                "sql": sql_query,
                                "response": human_response
                            })
                            
                            break  # Success, exit retry loop
                            
                        except Exception as e:
                            if attempt < st.session_state.processor.max_retries - 1:
                                st.warning(f"Retrying due to error: {str(e)}")
                                with st.spinner("Generating corrected query..."):
                                    sql_query = st.session_state.processor.handle_error(str(e), sql_query)
                            else:
                                st.error(f"Failed to execute query after {st.session_state.processor.max_retries} attempts: {str(e)}")
                                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display query history
        if st.session_state.query_history:
            with st.expander("Query History"):
                for i, item in enumerate(reversed(st.session_state.query_history)):
                    st.markdown(f"**Question {len(st.session_state.query_history)-i}:** {item['question']}")
                    st.code(item['sql'], language="sql")
                    st.markdown(f"**Answer:** {item['response']}")
                    st.divider()
    
    else:
        st.info("Please connect to a database using the sidebar.")

if __name__ == "__main__":
    main()