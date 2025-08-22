import os
import psycopg2
import pandas as pd
from typing import Dict, List, Any
from contextlib import contextmanager
import re
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from hdb_constraints import VALID_TOWNS, VALID_FLAT_TYPES, VALID_FLAT_MODELS, VALID_STOREY_RANGES

class HDBDatabase:
    """Database abstraction for HDB data queries and analysis"""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or os.getenv("NEON_DATABASE_URL")
        if not self.connection_string:
            raise ValueError("Database connection string not provided")
        
        self.llm = None  
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def get_schema_info(self, table_name: str = "hdb_resale_data") -> str:
        """Get detailed schema information for SQL generation"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            
            towns = VALID_TOWNS
            flat_types = VALID_FLAT_TYPES
            flat_models = VALID_FLAT_MODELS
            storey_ranges = VALID_STOREY_RANGES
            
            # Get date range
            cursor.execute(f"""
                SELECT MIN(month_date), MAX(month_date) FROM {table_name};
            """)
            date_range = cursor.fetchone()
            
            schema_info = f"""
TABLE: {table_name}

COLUMNS:
{chr(10).join([f"- {col[0]} ({col[1]}) {'NULL' if col[2] == 'YES' else 'NOT NULL'}" for col in columns])}

SAMPLE VALUES:
Towns: {', '.join(towns)}
Flat Types: {', '.join(flat_types)}
Flat Models: {', '.join(flat_models)}
Storey Ranges: {', '.join(storey_ranges)}
Date Range: {date_range[0]} to {date_range[1]}

NOTES:
- resale_price is in Singapore Dollars
- floor_area_sqm is in square meters
- remaining_lease is in years (decimal)
- month_date is better for date filtering than month string
- Use ILIKE for case-insensitive text matching
"""
            return schema_info
    
    def natural_language_to_sql(self, question: str, table_name: str = "hdb_resale_data") -> str:
        """Convert natural language question to SQL query"""
        
        if self.llm is None:
            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.1
            )
        
        schema_info = self.get_schema_info(table_name)
        
        sql_prompt = PromptTemplate(
            template="""
You are an expert SQL query generator for HDB resale data analysis.

DATABASE SCHEMA:
{schema_info}

QUESTION: "{question}"

IMPORTANT RULES:
1. Always use table name: {table_name}
2. For text matching, use ILIKE for case-insensitive search
3. For aggregations, use appropriate GROUP BY clauses
4. For price calculations, use AVG(), MIN(), MAX() functions
5. For counting, use COUNT(*) or COUNT(DISTINCT column)
6. For date filtering, use month_date column with DATE functions
7. Return only valid PostgreSQL syntax
8. Limit results to reasonable numbers (use LIMIT when appropriate)
9. For "most popular" queries, use COUNT and ORDER BY DESC
10. For "average" queries, use AVG() function

EXAMPLE QUERIES:
- "Most popular town": SELECT town, COUNT(*) as transaction_count FROM {table_name} GROUP BY town ORDER BY transaction_count DESC LIMIT 10;
- "Average price of 4 room flats": SELECT AVG(resale_price) as average_price FROM {table_name} WHERE flat_type = '4 ROOM';
- "Towns with highest prices": SELECT town, AVG(resale_price) as avg_price FROM {table_name} GROUP BY town ORDER BY avg_price DESC LIMIT 10;

Return ONLY the SQL query, no explanations or formatting:
""",
            input_variables=["question", "schema_info", "table_name"]
        )
        
        try:
            messages = [
                SystemMessage(content="Expert PostgreSQL query generator for HDB data. Return only valid SQL queries."),
                HumanMessage(content=sql_prompt.format(
                    question=question,
                    schema_info=schema_info,
                    table_name=table_name
                ))
            ]
            
            response = self.llm.invoke(messages)
            sql_query = response.content.strip()
            
            # Clean up the SQL query
            sql_query = re.sub(r'```sql\s*', '', sql_query)
            sql_query = re.sub(r'```\s*', '', sql_query)
            sql_query = sql_query.strip()
            
            return sql_query
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return None
    
    def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results as list of dictionaries"""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute(sql_query)
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                return results
                
            except Exception as e:
                print(f"Error executing query: {e}")
                print(f"Query: {sql_query}")
                return []
    
    def query_and_analyze(self, question: str) -> Dict[str, Any]:
        """Main method: Convert natural language to SQL, execute, and return results"""
        
        sql_query = self.natural_language_to_sql(question)
        
        if not sql_query:
            return {
                "error": "Could not generate SQL query from question",
                "question": question,
                "sql": None,
                "results": []
            }
        
        results = self.execute_query(sql_query)
        
        return {
            "question": question,
            "sql": sql_query,
            "results": results,
            "count": len(results)
        }
