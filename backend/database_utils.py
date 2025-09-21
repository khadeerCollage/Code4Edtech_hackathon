"""
Database configuration and utility functions for OMR Evaluation System
Provides database connection management and helper functions
"""
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from flask import current_app
import psycopg2
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration and connection management"""
    
    @staticmethod
    def get_db_url():
        """Get database URL with proper formatting"""
        db_url = os.getenv('DATABASE_URL')
        
        if not db_url:
            # Default to SQLite for development
            db_url = 'sqlite:///omr_evaluation.db'
            logger.warning("No DATABASE_URL found, using SQLite default")
        
        # Fix Heroku postgres:// URLs
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
            
        return db_url
    
    @staticmethod
    def get_engine_config():
        """Get SQLAlchemy engine configuration"""
        db_url = DatabaseConfig.get_db_url()
        
        config = {
            'echo': os.getenv('FLASK_ENV') == 'development',
            'pool_pre_ping': True,
        }
        
        # PostgreSQL specific configuration
        if db_url.startswith("postgresql://"):
            config.update({
                'poolclass': QueuePool,
                'pool_size': int(os.getenv('DB_POOL_SIZE', 10)),
                'pool_overflow': int(os.getenv('DB_POOL_OVERFLOW', 20)),
                'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', 3600)),
            })
        
        return config

def test_database_connection(db_url=None):
    """Test database connection"""
    if not db_url:
        db_url = DatabaseConfig.get_db_url()
    
    try:
        if db_url.startswith("postgresql://"):
            # Test PostgreSQL connection
            parsed = urlparse(db_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password,
                port=parsed.port or 5432
            )
            conn.close()
            logger.info("✅ PostgreSQL connection successful")
            return True
        elif db_url.startswith("sqlite://"):
            # Test SQLite connection
            engine = create_engine(db_url)
            connection = engine.connect()
            connection.close()
            logger.info("✅ SQLite connection successful")
            return True
        else:
            logger.error(f"❌ Unsupported database URL: {db_url}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def create_database_if_not_exists(db_url=None):
    """Create database if it doesn't exist (PostgreSQL only)"""
    if not db_url:
        db_url = DatabaseConfig.get_db_url()
    
    if not db_url.startswith("postgresql://"):
        logger.info("Database auto-creation only supported for PostgreSQL")
        return True
    
    try:
        parsed = urlparse(db_url)
        db_name = parsed.path[1:]
        
        # Connect to default postgres database
        admin_url = db_url.replace(f"/{db_name}", "/postgres")
        
        conn = psycopg2.connect(admin_url)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"✅ Created database: {db_name}")
        else:
            logger.info(f"✅ Database already exists: {db_name}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create database: {e}")
        return False

def initialize_database_schema(app):
    """Initialize database schema from SQL file"""
    try:
        schema_file = os.path.join(os.path.dirname(__file__), 'database_schema.sql')
        
        if not os.path.exists(schema_file):
            logger.warning(f"Schema file not found: {schema_file}")
            return False
        
        db_url = app.config['SQLALCHEMY_DATABASE_URI']
        
        if db_url.startswith("postgresql://"):
            # Execute PostgreSQL schema
            parsed = urlparse(db_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password,
                port=parsed.port or 5432
            )
            
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            cursor = conn.cursor()
            cursor.execute(schema_sql)
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("✅ Database schema initialized from SQL file")
            return True
        else:
            logger.info("Schema initialization from SQL file only supported for PostgreSQL")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize schema: {e}")
        return False

def get_database_info():
    """Get database information for debugging"""
    try:
        from flask import current_app
        from sqlalchemy import inspect
        
        db_url = current_app.config['SQLALCHEMY_DATABASE_URI']
        engine = current_app.extensions['sqlalchemy'].db.engine
        inspector = inspect(engine)
        
        tables = inspector.get_table_names()
        
        info = {
            'database_url': db_url.split('@')[0] + '@' + db_url.split('@')[1].split('/')[0] + '/***' if '@' in db_url else db_url,
            'engine': str(engine.name),
            'tables': tables,
            'table_count': len(tables)
        }
        
        # Get row counts for each table
        table_counts = {}
        for table in tables:
            try:
                result = engine.execute(f"SELECT COUNT(*) FROM {table}")
                count = result.scalar()
                table_counts[table] = count
            except:
                table_counts[table] = 'N/A'
        
        info['table_counts'] = table_counts
        return info
        
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {"error": str(e)}

# Utility functions for common database operations
def backup_database(output_file=None):
    """Backup database (PostgreSQL only)"""
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"omr_backup_{timestamp}.sql"
    
    try:
        db_url = DatabaseConfig.get_db_url()
        
        if not db_url.startswith("postgresql://"):
            logger.error("Database backup only supported for PostgreSQL")
            return False
        
        parsed = urlparse(db_url)
        
        cmd = [
            'pg_dump',
            '-h', parsed.hostname,
            '-p', str(parsed.port or 5432),
            '-U', parsed.username,
            '-d', parsed.path[1:],
            '-f', output_file,
            '--verbose'
        ]
        
        import subprocess
        os.environ['PGPASSWORD'] = parsed.password
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Database backup created: {output_file}")
            return True
        else:
            logger.error(f"❌ Backup failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Backup error: {e}")
        return False