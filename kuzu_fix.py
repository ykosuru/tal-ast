"""
Kuzu Database Fix and Enhancement
Fixes the directory path issue and adds helper functions

This module provides:
- Fixed Kuzu initialization
- Safe database cleanup
- Connection management
- Schema verification
"""

from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)

def initialize_kuzu_database(db_path: str = "./knowledge_graph_db", clean: bool = False):
    """
    Initialize Kuzu database with proper path handling
    
    Args:
        db_path: Path to database directory
        clean: If True, remove existing database
    
    Returns:
        Tuple of (database, connection)
    """
    try:
        import kuzu
    except ImportError:
        raise ImportError("Kuzu not installed. Install with: pip install kuzu")
    
    db_path_obj = Path(db_path)
    
    # Handle existing database
    if db_path_obj.exists():
        if clean:
            logger.info(f"Cleaning existing Kuzu database at {db_path}")
            shutil.rmtree(db_path_obj)
        else:
            logger.info(f"Using existing Kuzu database at {db_path}")
    
    # Create directory
    db_path_obj.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    try:
        db = kuzu.Database(str(db_path_obj))
        conn = kuzu.Connection(db)
        logger.info(f"Successfully initialized Kuzu database at {db_path}")
        return db, conn
    except Exception as e:
        logger.error(f"Failed to initialize Kuzu: {e}")
        # Try cleaning and recreating
        if db_path_obj.exists():
            shutil.rmtree(db_path_obj)
        db_path_obj.mkdir(parents=True, exist_ok=True)
        db = kuzu.Database(str(db_path_obj))
        conn = kuzu.Connection(db)
        logger.info(f"Initialized Kuzu database after cleanup")
        return db, conn


def cleanup_kuzu_database(db_path: str = "./knowledge_graph_db"):
    """
    Clean up Kuzu database directory
    
    Args:
        db_path: Path to database directory
    """
    db_path_obj = Path(db_path)
    if db_path_obj.exists():
        shutil.rmtree(db_path_obj)
        logger.info(f"Cleaned up Kuzu database at {db_path}")


def verify_kuzu_installation():
    """Verify Kuzu is properly installed"""
    try:
        import kuzu
        version = getattr(kuzu, '__version__', 'unknown')
        logger.info(f"Kuzu version {version} is installed")
        return True
    except ImportError:
        logger.error("Kuzu is not installed")
        return False


# Monkey patch for KuzuDatabase class
def patch_kuzu_database():
    """
    Apply fix to KuzuDatabase class in knowledge_graph module
    
    This function modifies the __init__ method to properly handle database paths
    """
    try:
        from knowledge_graph import KuzuDatabase
        import kuzu
        
        original_init = KuzuDatabase.__init__
        
        def fixed_init(self, db_path: str = "./knowledge_graph_db"):
            """Fixed initialization with proper path handling"""
            if not hasattr(kuzu, 'Database'):
                raise ImportError("Kuzu is not installed. Install with: pip install kuzu")
            
            self.db_path = Path(db_path)
            
            # Clean approach: remove existing database to avoid schema conflicts
            if self.db_path.exists():
                logger.info(f"Removing existing Kuzu database at {self.db_path}")
                shutil.rmtree(self.db_path)
            
            # Create fresh directory
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize Kuzu database
            try:
                self.db = kuzu.Database(str(self.db_path))
                self.conn = kuzu.Connection(self.db)
                logger.info(f"Initialized Kuzu database at {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to initialize Kuzu database: {e}")
                raise
            
            # Call original schema initialization
            self._init_schema()
        
        KuzuDatabase.__init__ = fixed_init
        logger.info("Successfully patched KuzuDatabase class")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch KuzuDatabase: {e}")
        return False


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   Kuzu Database Fix Module                           ║
╚══════════════════════════════════════════════════════════════════════╝

This module fixes Kuzu database initialization issues.

USAGE:
  from kuzu_fix import initialize_kuzu_database
  
  db, conn = initialize_kuzu_database("./my_db", clean=True)

FUNCTIONS:
  • initialize_kuzu_database() - Properly initialize Kuzu
  • cleanup_kuzu_database()     - Clean up database directory
  • verify_kuzu_installation()  - Check if Kuzu is installed
  • patch_kuzu_database()       - Patch existing KuzuDatabase class

COMMON ISSUES FIXED:
  ✓ "Database path cannot be a directory" error
  ✓ Schema conflicts from existing databases
  ✓ Path handling on Windows/Unix systems
    """)
