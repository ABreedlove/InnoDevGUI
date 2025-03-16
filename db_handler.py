# db_handler.py
import sqlite3
import json
from typing import Dict, List, Optional, Any
from pathlib import Path  # For handling file paths
import os  # For handling file paths
from datetime import datetime  # If you want to add timestamps


class DatabaseHandler:
    def __init__(self, db_path: str = "topology.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.setup_database()
    
    def setup_database(self):
        """Create necessary tables if they don't exist"""
        with self.conn:
            cursor = self.conn.cursor()
            
            # Device table (existing)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS devices (
                    node_id TEXT PRIMARY KEY,
                    name TEXT,
                    ip TEXT UNIQUE,
                    is_snmp BOOLEAN,
                    community TEXT,
                    write_community TEXT,
                    version TEXT,
                    position_x REAL,
                    position_y REAL,
                    simulation_mode BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Settings table (existing)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            # New table for SNMP values history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS snmp_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT,
                    oid TEXT,
                    value TEXT,
                    category TEXT,
                    field_name TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (node_id) REFERENCES devices(node_id)
                )
            ''')
            
            # New table for device states/conditions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS device_states (
                    node_id TEXT,
                    state_type TEXT,
                    state_value TEXT,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (node_id, state_type),
                    FOREIGN KEY (node_id) REFERENCES devices(node_id)
                )
            ''')
            
            # New table for device settings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS device_settings (
                    node_id TEXT,
                    oid TEXT,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (node_id, oid)
                )
            ''')
            
            # New table for viewport state
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS viewport_state (
                    id INTEGER PRIMARY KEY,
                    scale REAL,
                    position_x REAL,
                    position_y REAL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # New index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_snmp_values_node_oid 
                ON snmp_values(node_id, oid)
            ''')
            
            self.conn.commit()
    
    def save_snmp_value(self, node_id: str, oid: str, value: str, category: str, field_name: str):
        """Save a new SNMP value reading"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO snmp_values (node_id, oid, value, category, field_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (node_id, oid, value, category, field_name))
            self.conn.commit()
    
    def get_latest_value(self, node_id: str, oid: str) -> Optional[str]:
        """Get the most recent value for a specific OID"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT value FROM snmp_values
                WHERE node_id = ? AND oid = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (node_id, oid))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_value_history(self, node_id: str, oid: str, hours: int = 24) -> List[Dict]:
        """Get historical values for a specific OID"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT value, timestamp
                FROM snmp_values
                WHERE node_id = ? AND oid = ?
                AND timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            ''', (node_id, oid, f'-{hours} hours'))
            
            return [
                {'value': row[0], 'timestamp': row[1]}
                for row in cursor.fetchall()
            ]
    
    def update_device_state(self, node_id: str, state_type: str, state_value: str):
        """Update device state for visual representation"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO device_states (node_id, state_type, state_value, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (node_id, state_type, state_value))
            self.conn.commit()
    
    def get_device_state(self, node_id: str, state_type: str) -> Optional[str]:
        """Get current device state"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT state_value FROM device_states
                WHERE node_id = ? AND state_type = ?
            ''', (node_id, state_type))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_all_device_states(self, node_id: str) -> Dict[str, str]:
        """Get all states for a device"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT state_type, state_value FROM device_states
                WHERE node_id = ?
            ''', (node_id,))
            return dict(cursor.fetchall())
    
    def save_device_position(self, node_id: str, x: float, y: float):
        """Update device position"""
        print(f"Saving position for {node_id}: x={x}, y={y}")  # Debug
        with self.conn:
            cursor = self.conn.cursor()
            # First check if device exists
            cursor.execute('SELECT 1 FROM devices WHERE node_id = ?', (node_id,))
            if not cursor.fetchone():
                print(f"Device {node_id} not found in database, inserting...")  # Debug
                # If device doesn't exist, insert it
                cursor.execute('''
                    INSERT INTO devices 
                    (node_id, position_x, position_y, name, ip, is_snmp, community, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (node_id, x, y, "Unknown", "0.0.0.0", True, "public", "v2c"))
            else:
                # Update existing device
                cursor.execute('''
                    UPDATE devices
                    SET position_x = ?, position_y = ?
                    WHERE node_id = ?
                ''', (x, y, node_id))
            self.conn.commit()
    
    def get_device_position(self, node_id: str) -> Optional[Dict[str, float]]:
        """Get device position by node_id"""
        #print(f"Getting position for {node_id}")  # Debug
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT position_x, position_y
                FROM devices
                WHERE node_id = ?
            ''', (node_id,))
            result = cursor.fetchone()
            
            if result:
                position = {'x': result[0], 'y': result[1]}
                #print(f"Found position for {node_id}: {position}")  # Debug
                return position
            print(f"No position found for {node_id}")  # Debug
            return None
    
    def save_device(self, device_data: Dict):
        """Save or update device"""
        print(f"Saving device to database: {device_data}")  # Debug
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO devices
                (node_id, name, ip, is_snmp, community, write_community, version, position_x, position_y, simulation_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                device_data['node_id'],
                device_data['name'],
                device_data['ip'],
                device_data['is_snmp'],
                device_data['community'],
                device_data.get('write_community', 'private'),
                device_data['version'],
                device_data['position']['x'],
                device_data['position']['y'],
                device_data.get('simulation_mode', False)
            ))
            self.conn.commit()
            print(f"Saved device {device_data['name']} with position {device_data['position']}")  # Debug

    
    def get_all_devices(self) -> List[Dict]:
        """Get all devices with their positions"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT node_id, name, ip, is_snmp, community, write_community, version, position_x, position_y, simulation_mode
                FROM devices
            ''')
            results = cursor.fetchall()
            
            devices = []
            for row in results:
                devices.append({
                    'node_id': row[0],
                    'name': row[1],
                    'ip': row[2],
                    'is_snmp': bool(row[3]),
                    'community': row[4],
                    'write_community': row[5],
                    'version': row[6],
                    'position': {'x': row[7], 'y': row[8]},
                    'simulation_mode': bool(row[9])
                })
            return devices
    
    def save_background_image(self, image_data: str, dimensions: Dict):
        """Save background image data"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value)
                VALUES (?, ?), (?, ?)
            ''', (
                'background_image', image_data,
                'background_dimensions', json.dumps(dimensions)
            ))
            self.conn.commit()
    
    def get_background_image(self) -> Optional[Dict]:
        """Get background image data"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT key, value FROM settings
                WHERE key IN ('background_image', 'background_dimensions')
            ''')
            results = dict(cursor.fetchall())
            
            if 'background_image' in results:
                return {
                    'image': results['background_image'],
                    'dimensions': json.loads(results.get('background_dimensions', '{}'))
                }
            return None
    
    def clear_all_data(self):
        """Clear all data from database"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM devices')
            cursor.execute('DELETE FROM settings')
            self.conn.commit()

    def remove_device(self, node_id: str):
        """Remove a device from the database"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM devices WHERE node_id = ?', (node_id,))
            self.conn.commit()

    def clean_database(self):
        """Clean up database by removing duplicates and invalid entries"""
        with self.conn:
            cursor = self.conn.cursor()
            
            # Remove devices with null or empty IPs
            cursor.execute('''
                DELETE FROM devices 
                WHERE ip IS NULL OR ip = '' OR ip = '0.0.0.0'
            ''')
            
            # Remove duplicate IPs, keeping the most recently updated one
            cursor.execute('''
                DELETE FROM devices 
                WHERE rowid NOT IN (
                    SELECT MAX(rowid)
                    FROM devices
                    GROUP BY ip
                )
            ''')
            
            # Clean up orphaned device states
            cursor.execute('''
                DELETE FROM device_states
                WHERE node_id NOT IN (SELECT node_id FROM devices)
            ''')
            
            # Clean up orphaned SNMP values
            cursor.execute('''
                DELETE FROM snmp_values
                WHERE node_id NOT IN (SELECT node_id FROM devices)
            ''')
            
            self.conn.commit()
    
    def save_snmp_value(self, node_id: str, oid: str, value: str, category: str, field_name: str):
        """Save a new SNMP value reading"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO snmp_values (node_id, oid, value, category, field_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (node_id, oid, value, category, field_name))
            self.conn.commit()
    
    def save_viewport_state(self, scale: float, position: Dict[str, float]) -> None:
        """Save the viewport scale and position to the database"""
        with self.conn:
            cursor = self.conn.cursor()
            # First delete any existing viewport state
            cursor.execute('DELETE FROM viewport_state')
            # Then insert the new state
            cursor.execute('''
                INSERT INTO viewport_state (scale, position_x, position_y)
                VALUES (?, ?, ?)
            ''', (scale, position['x'], position['y']))

    def get_viewport_state(self) -> Optional[Dict[str, Any]]:
        """Get the saved viewport state from the database"""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('SELECT scale, position_x, position_y FROM viewport_state')
            result = cursor.fetchone()
            
            if result:
                scale, pos_x, pos_y = result
                return {
                    'scale': scale,
                    'position': {'x': pos_x, 'y': pos_y}
                }
            return None