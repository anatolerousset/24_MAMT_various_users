"""
User management system for multi-user RAG application
"""
import hashlib
import uuid
import time
import logging
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime, timedelta

_log = logging.getLogger(__name__)

class UserManager:
    """Manages user sessions and their associated resources"""
    
    def __init__(self):
        self.active_users: Dict[str, Dict] = {}
        self.session_timeout = 3600  # 1 hour timeout
        self.cleanup_interval = 300   # 5 minutes cleanup interval
        self.last_cleanup = time.time()
    
    def create_user_session(self, user_identifier: Optional[str] = None) -> str:
        """
        Create a new user session
        
        Args:
            user_identifier: Optional user identifier (email, username, etc.)
            
        Returns:
            User session ID
        """
        # Generate unique session ID
        if user_identifier:
            # Use hash of identifier + timestamp for reproducible but unique sessions
            session_data = f"{user_identifier}_{int(time.time())}"
            session_id = hashlib.sha256(session_data.encode()).hexdigest()[:16]
        else:
            # Generate random session ID
            session_id = str(uuid.uuid4()).replace('-', '')[:16]
        
        # Create user session data
        user_data = {
            'session_id': session_id,
            'user_identifier': user_identifier,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'dce_collection': f"dce_user_{session_id}",
            'temp_image_folder': f"temp_images_{session_id}",
            'public_image_folder': f"public_{session_id}",
            'processed_files': [],
            'active': True
        }
        
        self.active_users[session_id] = user_data
        
        # Cleanup old sessions periodically
        self._cleanup_expired_sessions()
        
        _log.info(f"Created user session: {session_id} for {user_identifier or 'anonymous'}")
        return session_id
    
    def get_user_session(self, session_id: str) -> Optional[Dict]:
        """Get user session data"""
        if session_id in self.active_users:
            user_data = self.active_users[session_id]
            
            # Check if session is expired
            if self._is_session_expired(user_data):
                self.cleanup_user_session(session_id)
                return None
            
            # Update last activity
            user_data['last_activity'] = datetime.now()
            return user_data
        
        return None
    
    def update_user_activity(self, session_id: str):
        """Update user's last activity timestamp"""
        if session_id in self.active_users:
            self.active_users[session_id]['last_activity'] = datetime.now()
    
    def get_user_dce_collection(self, session_id: str) -> str:
        """Get user's DCE collection name"""
        user_data = self.get_user_session(session_id)
        if user_data:
            return user_data['dce_collection']
        return f"dce_user_{session_id}"  # Fallback
    
    def get_user_temp_image_folder(self, session_id: str) -> str:
        """Get user's temporary image folder name"""
        user_data = self.get_user_session(session_id)
        if user_data:
            return user_data['temp_image_folder']
        return f"temp_images_{session_id}"  # Fallback
    
    def get_user_public_image_folder(self, session_id: str) -> str:
        """Get user's public image folder name"""
        user_data = self.get_user_session(session_id)
        if user_data:
            return user_data['public_image_folder']
        return f"public_{session_id}"  # Fallback
    
    def add_processed_file(self, session_id: str, filename: str):
        """Add a processed file to user's session"""
        user_data = self.get_user_session(session_id)
        if user_data:
            user_data['processed_files'].append({
                'filename': filename,
                'processed_at': datetime.now()
            })
    
    def get_processed_files(self, session_id: str) -> List[Dict]:
        """Get list of processed files for user"""
        user_data = self.get_user_session(session_id)
        if user_data:
            return user_data.get('processed_files', [])
        return []
    
    def cleanup_user_session(self, session_id: str):
        """Cleanup a specific user session"""
        if session_id in self.active_users:
            user_data = self.active_users[session_id]
            
            # Mark as inactive
            user_data['active'] = False
            
            # Clean up user's temporary files
            self._cleanup_user_files(user_data)
            
            # Remove from active users
            del self.active_users[session_id]
            
            _log.info(f"Cleaned up user session: {session_id}")
    
    def _cleanup_user_files(self, user_data: Dict):
        """Clean up user's temporary files and folders"""
        try:
            # Clean up temporary image folder
            temp_folder = Path("temp") / user_data['temp_image_folder']
            if temp_folder.exists():
                import shutil
                shutil.rmtree(temp_folder)
                _log.debug(f"Cleaned up temp folder: {temp_folder}")
            
            # Clean up public image folder
            public_folder = Path("public") / user_data['public_image_folder']
            if public_folder.exists():
                import shutil
                shutil.rmtree(public_folder)
                _log.debug(f"Cleaned up public folder: {public_folder}")
                
        except Exception as e:
            _log.error(f"Error cleaning up user files: {e}")
    
    def _is_session_expired(self, user_data: Dict) -> bool:
        """Check if a user session is expired"""
        last_activity = user_data.get('last_activity', datetime.min)
        return (datetime.now() - last_activity).total_seconds() > self.session_timeout
    
    def _cleanup_expired_sessions(self):
        """Clean up expired user sessions"""
        current_time = time.time()
        
        # Only cleanup if enough time has passed
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        expired_sessions = []
        for session_id, user_data in self.active_users.items():
            if self._is_session_expired(user_data):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.cleanup_user_session(session_id)
        
        if expired_sessions:
            _log.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        self.last_cleanup = current_time
    
    def list_active_users(self) -> List[Dict]:
        """List all active user sessions"""
        active_users = []
        for session_id, user_data in self.active_users.items():
            if not self._is_session_expired(user_data):
                active_users.append({
                    'session_id': session_id,
                    'user_identifier': user_data.get('user_identifier'),
                    'created_at': user_data['created_at'].isoformat(),
                    'last_activity': user_data['last_activity'].isoformat(),
                    'dce_collection': user_data['dce_collection'],
                    'processed_files_count': len(user_data.get('processed_files', []))
                })
        
        return active_users
    
    def get_stats(self) -> Dict:
        """Get user management statistics"""
        self._cleanup_expired_sessions()
        
        return {
            'active_users': len(self.active_users),
            'total_processed_files': sum(
                len(user_data.get('processed_files', [])) 
                for user_data in self.active_users.values()
            ),
            'session_timeout_hours': self.session_timeout / 3600,
            'cleanup_interval_minutes': self.cleanup_interval / 60
        }


# Global user manager instance
_user_manager = None

def get_user_manager() -> UserManager:
    """Get or create the global user manager instance"""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager