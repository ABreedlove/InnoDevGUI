import streamlit as st
import signal
import time

# Page config must be the absolute first Streamlit command
st.set_page_config(
    page_title="SNMP Device Manager",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Then initialize session state
if 'devices' not in st.session_state:
    st.session_state['devices'] = []
if 'background_image' not in st.session_state:
    st.session_state['background_image'] = None
if 'background_dimensions' not in st.session_state:
    st.session_state['background_dimensions'] = {'width': 800, 'height': 800}
if 'loaded_from_file' not in st.session_state:
    st.session_state['loaded_from_file'] = False
if 'selected_device_id' not in st.session_state:
    st.session_state.selected_device_id = None
if 'previous_states' not in st.session_state:
    st.session_state.previous_states = {}
if 'expander_states' not in st.session_state:
    st.session_state.expander_states = {}
if 'last_viewport_save' not in st.session_state:
    st.session_state.last_viewport_save = 0
if 'viewport_debounce_ms' not in st.session_state:
    st.session_state.viewport_debounce_ms = 500  # Debounce time in milliseconds
if 'position_update_queue' not in st.session_state:
    st.session_state.position_update_queue = {}
if 'last_position_update' not in st.session_state:
    st.session_state.last_position_update = time.time()
if 'is_polling' not in st.session_state:
    st.session_state.is_polling = False

from streamlit_vis import vis_network
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.elements import layouts
from streamlit_autorefresh import st_autorefresh


import streamlit.components.v1 as components
# First, update the import
#from streamlit_cytoscapejs import CytoscapeComponent, create_cytoscape_component
#from streamlit_cytoscapejs import st_cytoscapejs as st_cytoscope
import json
import asyncio
from asyncio import create_task, gather
from typing import List, Dict, Optional, Union, Tuple, Any
import ipaddress
import base64
from PIL import Image
import io
import os
from datetime import datetime, timedelta
from collections import deque


from pysnmp.hlapi.v3arch.asyncio import (
    SnmpEngine, CommunityData, UdpTransportTarget,
    ContextData, ObjectType, ObjectIdentity, get_cmd, set_cmd
)

from pysnmp.proto.rfc1902 import (
    Integer, OctetString, IpAddress,
    Counter32, Counter64, Gauge32,
    TimeTicks, Unsigned32
)

from db_handler import DatabaseHandler  # Import our custom database handler
import random


# SNMP OID Constants
GENERAL_OIDS = {
    'Serial Number': '1.3.6.1.4.1.60811.1.1.1',
    'Mac Address': '1.3.6.1.4.1.60811.1.1.2',
    'GMT Offset': '1.3.6.1.4.1.60811.1.1.3',
    'Organization Name': '1.3.6.1.4.1.60811.1.1.5',
    'Door Label': '1.3.6.1.4.1.60811.1.1.6',
    'IP Address': '1.3.6.1.4.1.60811.1.1.7',
    'IP Type': '1.3.6.1.4.1.60811.1.1.8',
    'Silent Flag': "1.3.6.1.4.1.60811.1.1.9",
}

DOOR_OIDS = {
    'Door Trigger Set Point': '1.3.6.1.4.1.60811.1.1.10',
    'Reset Delay': '1.3.6.1.4.1.60811.1.1.11',
    'Door Live Status': '1.3.6.1.4.1.60811.1.1.12',
    'Triggered Status': '1.3.6.1.4.1.60811.1.1.13',
    'Live Triggered Timer': '1.3.6.1.4.1.60811.1.1.14',
    'Last Open Timestamp': '1.3.6.1.4.1.60811.1.1.15',
    'Last Triggered Timestamp': '1.3.6.1.4.1.60811.1.1.16'
}


TEMPERATURE_OIDS = {
    'Live Temperature': '1.3.6.1.4.1.60811.1.1.54',
    'Triggered Status [Temp.]': '1.3.6.1.4.1.60811.1.1.26',
    'High Temp. Trigger Set Point': '1.3.6.1.4.1.60811.1.1.28',
    'Low Temp. Trigger Set Point': '1.3.6.1.4.1.60811.1.1.29',
    'Unit of Measurement': '1.3.6.1.4.1.60811.1.1.32',
    'Last Triggered Timestamp [Temp.]': '1.3.6.1.4.1.60811.1.1.43',
    'Live Triggered Timer [Temp.]': '1.3.6.1.4.1.60811.1.1.40',
    'Reset Delay (Shared)': '1.3.6.1.4.1.60811.1.1.42',
    'Highest Temperature': '1.3.6.1.4.1.60811.1.1.51',
    'Lowest Temperature': '1.3.6.1.4.1.60811.1.1.50'
}

HUMIDITY_OIDS = {
    'Live Humidity': '1.3.6.1.4.1.60811.1.1.55',
    'Triggered Status [Humid.]': '1.3.6.1.4.1.60811.1.1.27',
    'High Humid. Trigger Set Point': '1.3.6.1.4.1.60811.1.1.30',
    'Low Humid. Trigger Set Point': '1.3.6.1.4.1.60811.1.1.31',
    'Live Triggered Timer [Humid.]': '1.3.6.1.4.1.60811.1.1.41',
    'Reset Delay (Shared)': '1.3.6.1.4.1.60811.1.1.42',
    'Last Triggered Timestamp [Humid.]': '1.3.6.1.4.1.60811.1.1.44',
    'Highest Humidity': '1.3.6.1.4.1.60811.1.1.53',
    'Lowest Humidity': '1.3.6.1.4.1.60811.1.1.52'
}

RATEOFCHANGE_OIDS = {
    'RoC Trigger Status [Temp.]': '1.3.6.1.4.1.60811.1.1.62',
    'RoC Trigger Status [Humid.]': '1.3.6.1.4.1.60811.1.1.63',
    'Current RoC [Temp.]': '1.3.6.1.4.1.60811.1.1.59',
    'Current RoC [Humid.]': '1.3.6.1.4.1.60811.1.1.60',
    'Maximum Rate of Change [Temp.]': '1.3.6.1.4.1.60811.1.1.56',
    'Maximum Rate of Change [Humid.]': '1.3.6.1.4.1.60811.1.1.57',
    'RoC Unit of Measure': '1.3.6.1.4.1.60811.1.1.58',
    'RoC # of Iterations Until Reading': '1.3.6.1.4.1.60811.1.1.61',
    'Last Triggered RoC Timestamp [Temp.]': '1.3.6.1.4.1.60811.1.1.64',
    'Last Triggered RoC Timestamp [Humid.]': '1.3.6.1.4.1.60811.1.1.65'
}


EDITABLE_FIELDS = {
    'Door Trigger Set Point': True,
    'Serial Number': True,
    'Mac Address': False,
    'GMT Offset': True,
    'Daylights Savings Offset': True,
    'Organization Name': True,
    'Door Label': True,
    'IP Address': True,
    'IP Type': True,
    'Silent Flag': True,
    'Trigger Set Point': True,
    'Reset Delay': True,
    'Door Live Status': False,
    'Triggered Status': True,
    'Triggered Status Overall': True,
    'Live Triggered Timer': False,
    'Last Open Timestamp': False,
    'Last Triggered Timestamp': False,
    'Live Temperature': False,
    'Triggered Status [Temp.]': True,
    'High Temp. Trigger Set Point': True,
    'Low Temp. Trigger Set Point': True,
    'Unit of Measurement': True,
    'Live Triggered Timer [Temp.]': False,
    'Reset Delay (Shared)': True,
    'Last Triggered Timestamp [Temp.]': False,
    'Highest Temperature': False,
    'Lowest Temperature': False,
    'Live Humidity': False,
    'Triggered Status [Humid.]': True,
    'High Humid. Trigger Set Point': True,
    'Low Humid. Trigger Set Point': True,
    'Live Triggered Timer [Humid.]': False,
    'Last Triggered Timestamp [Humid.]': False,
    'Highest Humidity': False,
    'Lowest Humidity': False,
    'RoC Trigger Status [Temp.]': True,
    'RoC Trigger Status [Humid.]': True,
    'Current RoC [Temp.]': False,
    'Current RoC [Humid.]': False,
    'Maximum Rate of Change [Temp.]': True,
    'Maximum Rate of Change [Humid.]': True,
    'RoC Unit of Measure': True,
    'RoC # of Iterations Until Reading': True,
    'Last Triggered RoC Timestamp [Temp.]': False,
    'Last Triggered RoC Timestamp [Humid.]': False
}

# At the top of your gui.py, initialize the database handler
db = DatabaseHandler()

    
class Device:
    def __init__(self, ip: str, name: str = "Unknown", is_snmp: bool = True, 
                 community: str = "public", version: str = "v2c", position: Optional[Dict] = None,
                 write_community: str = "private", simulation_mode: bool = False):
        self.ip = ip
        self.name = name
        self.is_snmp = is_snmp
        self.community = community
        self.write_community = write_community
        self.version = version
        self.position = position or {'x': 100, 'y': 100}
        self.node_id = self.generate_node_id()
        self.simulation_mode = simulation_mode
        self.last_simulated_values = {}  # Store simulated values in memory
        
    def generate_node_id(self) -> str:
        return f"device_{self.ip.replace('.', '_')}"
        
    def to_dict(self) -> Dict:
        return {
            'ip': self.ip,
            'name': self.name,
            'is_snmp': self.is_snmp,
            'community': self.community,
            'write_community': self.write_community,
            'version': self.version,
            'position': self.position,
            'node_id': self.node_id,
            'simulation_mode': self.simulation_mode,
            'last_simulated_values': self.last_simulated_values  # Save simulation values
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Device':
        device = cls(
            ip=data['ip'],
            name=data['name'],
            is_snmp=data.get('is_snmp', True),
            community=data.get('community', 'public'),
            version=data.get('version', 'v2c'),
            position=data.get('position', {'x': 100, 'y': 100}),
            write_community=data.get('write_community', 'private'),
            simulation_mode=data.get('simulation_mode', False)
        )
        # Restore simulation values
        device.last_simulated_values = data.get('last_simulated_values', {})
        return device

    def update_position(self, x: float, y: float):
        """Update device position"""
        self.position = {'x': x, 'y': y}


class SNMPPoller:
    """Class to manage SNMP polling operations for devices"""
    def __init__(self):
        self.polling_interval = 60  # Default 60 seconds
        self.last_poll_time = {}  # Track last poll time per device
        self.pending_updates = []  # Queue of updates to be processed
        self.next_poll_time = None  # Track next scheduled poll
        self.threshold_cache = {}  # Store {device_id: {oid: (value, timestamp)}}
        self.threshold_ttl = 3600  # 1 hour TTL for thresholds

    async def get_cached_threshold(self, device: Device, oid: str, category: str, field_name: str) -> str:
        """Get threshold value from cache or fetch from device"""
        current_time = time.time()
        cache_key = (device.node_id, oid)
        
        # Check if we have a valid cached value
        if cache_key in self.threshold_cache:
            value, timestamp = self.threshold_cache[cache_key]
            if current_time - timestamp < self.threshold_ttl:
                return value
        
        # No valid cache, fetch new value
        value = await snmp_get(device.ip, oid, device.community)
        self.threshold_cache[cache_key] = (value, current_time)
        return value

    def should_poll_device(self, device_id: str) -> bool:
        """Check if device should be polled based on interval"""
        last_poll = self.last_poll_time.get(device_id, 0)
        current_time = time.time()
        return current_time - last_poll >= self.polling_interval
    
    async def poll_device(self, device: Device) -> None:
        """Poll a single device for updates and calculate states"""
        try:
            device_updates = {
                'device_id': device.node_id,
                'snmp_updates': [],
                'state_updates': [],
                'timestamp': time.time()
            }
            
            # Define all OIDs to poll for this device
            oid_batch = [
                (TEMPERATURE_OIDS['Live Temperature'], 'Live Temperature', 'Temperature'),
                (HUMIDITY_OIDS['Live Humidity'], 'Live Humidity', 'Humidity'),
                (DOOR_OIDS['Triggered Status'], 'Triggered Status', 'Door')
            ]
            
            # Create tasks for all OID queries at once
            query_tasks = [
                snmp_get(device.ip, oid, device.community)
                for oid, _, _ in oid_batch
            ]
            
            # Execute all SNMP queries concurrently
            try:
                values = await asyncio.gather(*query_tasks)
                
                # Process results in parallel
                state_tasks = []
                for (oid, field_name, category), value in zip(oid_batch, values):
                    if isinstance(value, str) and not value.startswith("Error"):
                        state_tasks.append(
                            calculate_device_states(device, oid, field_name, category, value, self)
                        )
                
                # Gather all state calculations
                if state_tasks:
                    state_results = await asyncio.gather(*state_tasks)
                    
                    # Combine all updates
                    for result in state_results:
                        if isinstance(result, dict) and not result.get('error'):
                            device_updates['snmp_updates'].extend(result.get('snmp_updates', []))
                            device_updates['state_updates'].extend(result.get('state_updates', []))
                    
                    # Only add to pending updates if we have actual updates
                    if device_updates['snmp_updates'] or device_updates['state_updates']:
                        self.pending_updates.append(device_updates)
                        self.last_poll_time[device.node_id] = time.time()
                
            except Exception as e:
                print(f"Error in batch SNMP queries for device {device.name}: {str(e)}")
                
        except Exception as e:
            print(f"Error polling device {device.name}: {str(e)}")

    async def _poll_single_oid(self, device: Device, oid: str, field_name: str, category: str) -> Dict:
        """Poll a single OID and calculate its states"""
        try:
            # Get the SNMP value
            value = await snmp_get(device.ip, oid, device.community)
            
            # Only process if we got a valid value
            if isinstance(value, str) and not value.startswith("Error"):
                # Calculate states based on the value
                states = await calculate_device_states(device, oid, field_name, category, value, self)
                return states
            
            return {'error': True, 'snmp_updates': [], 'state_updates': []}
            
        except Exception as e:
            print(f"Error polling OID {oid} for device {device.name}: {str(e)}")
            return {'error': True, 'snmp_updates': [], 'state_updates': []}
    
    
def initialize_polling_state():
    """Initialize polling-related session state"""
    if 'snmp_poller' not in st.session_state:
        st.session_state.snmp_poller = SNMPPoller()
    if 'is_polling' not in st.session_state:
        st.session_state.is_polling = False

def add_polling_controls():
    """Add polling controls to sidebar with improved timing logic"""
    initialize_polling_state()
    poller = st.session_state.snmp_poller
    
    # Initialize timing states if needed
    if 'last_polling_check' not in st.session_state:
        st.session_state.last_polling_check = time.time()
    if 'next_poll_time' not in st.session_state:
        st.session_state.next_poll_time = time.time()
    if 'poll_iteration' not in st.session_state:  # Add counter to force updates
        st.session_state.poll_iteration = 0
        
    with st.sidebar:
        st.markdown("---")
        st.subheader("SNMP Polling Settings")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            interval = st.number_input(
                "Polling Interval (seconds)",
                min_value=5,
                max_value=3600,
                value=poller.polling_interval,
                key=f"interval_{st.session_state.poll_iteration}"  # Dynamic key
            )
        
        enabled = st.toggle(
            "Enable Background Polling",
            value=st.session_state.is_polling,
            key=f"toggle_{st.session_state.poll_iteration}"  # Dynamic key
        )
        current_time = time.time()
        
        if enabled and current_time >= st.session_state.next_poll_time:
            # Use a longer refresh interval to reduce UI updates
            refresh_count = st_autorefresh(interval=max(interval * 1000, 5000), key="refresh_counter")
            st.session_state.is_polling = True
            
            poller.polling_interval = interval
            
            current_time = time.time()
            
            # Show time until next poll
            time_until_next = max(0, st.session_state.next_poll_time - current_time)
            st.caption(f"Next poll in: {int(time_until_next)} seconds")
            
            # Check if it's time to poll
            if current_time >= st.session_state.next_poll_time:
                needs_poll = False
                for device in st.session_state.devices:
                    if device.is_snmp and poller.should_poll_device(device.node_id):
                        needs_poll = True
                        break
                
                if needs_poll:
                    async def run_polls():
                        tasks = []
                        for device in st.session_state.devices:
                            if device.is_snmp and poller.should_poll_device(device.node_id):
                                tasks.append(poller.poll_device(device))
                        if tasks:
                            await asyncio.gather(*tasks)
                            return True
                        return False
                    
                    try:
                        # First run the polls
                        updates_made = asyncio.run(run_polls())
                        
                        # Then process any pending updates
                        if poller.pending_updates:
                            batch_updates_made = apply_batch_updates(poller.pending_updates, poller)
                            poller.pending_updates.clear()
                            
                            # Increment counter if either polls or updates made changes
                            if updates_made or batch_updates_made:
                                st.session_state.poll_iteration += 1
                    except Exception as e:
                        st.error(f"Error during polling: {str(e)}")
                    finally:
                        # Always set next poll time
                        st.session_state.next_poll_time = current_time + interval
            
            # Status display with reduced updates
            status_container = st.empty()
            with status_container:
                st.success(f"Polling active ({interval}s interval)")
                if st.checkbox("Show Last Poll Times", key=f"show_times_{st.session_state.poll_iteration}"):
                    current_time = time.time()
                    for device in st.session_state.devices:
                        last_poll = poller.last_poll_time.get(device.node_id)
                        if last_poll:
                            time_since = int(current_time - last_poll)
                            st.text(f"{device.name}: {time_since}s ago")
                        else:
                            st.text(f"{device.name}: Never")
        else:
            st.session_state.is_polling = True
            if 'next_poll_time' in st.session_state:
                del st.session_state.next_poll_time
            st.session_state.poll_iteration = 0  # Reset iteration counter
            
        # Show polling status and times
        if st.session_state.is_polling:
            st.success(f"Polling active ({interval}s interval)")
            
            # Add dynamic elements to force regular updates
            placeholder = st.empty()
            with placeholder.container():
                current_time = time.time()
                
                # Force update of time display
                st.markdown(f"""
                    <div style='display: none'>{current_time}</div>
                    Last update: {time.strftime('%H:%M:%S')}
                """, unsafe_allow_html=True)
        else:
            st.info("Polling inactive")
            
            

def generate_simulated_value(device: Device, oid: str, field_name: str) -> str:
    """Generate realistic simulated values for SNMP OIDs without database operations"""
    if oid not in device.last_simulated_values:
        # Initialize with sensible defaults
        if field_name == 'Live Temperature':
            device.last_simulated_values[oid] = "23.5"  # Room temperature
        elif field_name == 'Live Humidity':
            device.last_simulated_values[oid] = "45.0"  # Moderate humidity
        elif field_name == 'Triggered Status':
            device.last_simulated_values[oid] = "0"  # Door closed
        else:
            device.last_simulated_values[oid] = "0"
    
    last_value = float(device.last_simulated_values[oid])
    
    if field_name == 'Live Temperature':
        change = random.uniform(-2.0, 2.0)
        new_value = max(10.0, min(35.0, last_value + change))
    elif field_name == 'Live Humidity':
        change = random.uniform(-3.0, 3.0)
        new_value = max(20.0, min(80.0, last_value + change))
    elif field_name == 'Triggered Status':
        if random.random() < 0.3:  # 30% chance to change state
            new_value = 1 if last_value == 0 else 0
        else:
            new_value = last_value
    else:
        # For threshold values, return stored value or default
        return device.last_simulated_values[oid]
    
    device.last_simulated_values[oid] = str(round(new_value, 1))
    return device.last_simulated_values[oid]

async def snmp_get(ip: str, oid: str, community: str = 'public') -> str:
    """Perform SNMP GET operation with simulation mode support but no database operations"""
    device = next((d for d in st.session_state['devices'] if d.ip == ip), None)
    if device and device.simulation_mode:
        try:
            # Get field name for simulation
            field_name = next((name for name, o in {**GENERAL_OIDS, **DOOR_OIDS, **TEMPERATURE_OIDS, **HUMIDITY_OIDS}.items() if o == oid), "Unknown")
            
            # For essential readings that change
            if oid in [TEMPERATURE_OIDS['Live Temperature'], 
                      HUMIDITY_OIDS['Live Humidity'], 
                      DOOR_OIDS['Triggered Status']]:
                return generate_simulated_value(device, oid, field_name)
            
            # For threshold and configuration values
            # First check memory cache
            if oid in device.last_simulated_values:
                print("FOUND SIMULATED VALUE")
                return device.last_simulated_values[oid]
                
            # Return sensible defaults for thresholds
            if oid in TEMPERATURE_OIDS.values():
                if 'High' in field_name:
                    return "35"
                elif 'Low' in field_name:
                    return "10"
            elif oid in HUMIDITY_OIDS.values():
                if 'High' in field_name:
                    return "80"
                elif 'Low' in field_name:
                    return "20"
            elif oid in DOOR_OIDS.values():
                if 'Trigger' in field_name:
                    return "60"
            return "0"

        except Exception as e:
            print(f"Error in simulation mode for {oid}: {str(e)}")
            return f"Error: {str(e)}"
    
    # Normal SNMP GET operation
    try:
        transportObj = await UdpTransportTarget.create((ip, 161), timeout=2, retries=1)
        error_indication, error_status, error_index, var_binds = await get_cmd(
            SnmpEngine(),
            CommunityData(community, mpModel=0),
            transportObj,
            ContextData(),
            ObjectType(ObjectIdentity(oid))
        )
        
        if error_indication or error_status:
            return f"Error: {error_indication or error_status}"
        
        return var_binds[0][1].prettyPrint()
    except Exception as e:
        return f"Error: {str(e)}"
    
    
# Enhanced SNMP SET function with better error handling
async def snmp_set(ip: str, oid: str, value: str, community: str = 'private') -> Tuple[bool, str]:
    """
    Enhanced SNMP SET operation with proper type handling
    
    Args:
        ip: Device IP address
        oid: SNMP OID to set
        value: Value to set (as string)
        community: SNMP write community string
        
    Returns:
        Tuple[bool, str]: (success, message)
    """
    device = next((d for d in st.session_state['devices'] if d.ip == ip), None)
    if device and device.simulation_mode:
        try:
            # Convert value to appropriate type
            snmp_value = determine_snmp_type(oid, value)
            
            category = 'SET'
            field_name = 'Simulated SET'
            
            if oid in TEMPERATURE_OIDS.values():
                category = 'Temperature'
                keys = [key for key, val in TEMPERATURE_OIDS.items() if val == oid]
                field_name = keys[0]
            elif oid in HUMIDITY_OIDS.values():
                category = 'Humidity'
                keys = [key for key, val in HUMIDITY_OIDS.items() if val == oid]
                field_name = keys[0]
            elif oid in DOOR_OIDS.values():    
                category = 'Door'
                keys = [key for key, val in DOOR_OIDS.items() if val == oid]
                field_name = keys[0]
            elif oid in GENERAL_OIDS.values():
                category = 'General'
                keys = [key for key, val in GENERAL_OIDS.items() if val == oid]
                field_name = keys[0]
            
            
            
            # Update the device's simulated values
            device.last_simulated_values[oid] = str(snmp_value)
            
            return True, "Success"
            
        except Exception as e:
            return False, f"Error in simulation mode: {str(e)}"

    
    # Normal SNMP SET for non-simulation mode
    try:
        # Convert value to appropriate SNMP type
        snmp_value = determine_snmp_type(oid, value)
        
        transportObj = await UdpTransportTarget.create((ip, 161), timeout=3, retries=2)
        error_indication, error_status, error_index, var_binds = await set_cmd(
            SnmpEngine(),
            CommunityData(community, mpModel=0),
            transportObj,
            ContextData(),
            ObjectType(ObjectIdentity(oid), snmp_value)
        )
        
        if error_indication:
            return False, f"Error: {error_indication}"
        elif error_status:
            return False, f"Error: {error_status.prettyPrint()}"
        
        # Verify the set was successful
        error_indication, error_status, error_index, var_binds = await get_cmd(
            SnmpEngine(),
            CommunityData(community, mpModel=0),
            transportObj,
            ContextData(),
            ObjectType(ObjectIdentity(oid))
        )
        
        if error_indication or error_status:
            return False, "Error: Set verification failed"
        
        # Compare set value (account for type differences)
        set_value = str(var_binds[0][1])
        if isinstance(snmp_value, IpAddress):
            if set_value != value:
                return False, "Error: Set verification failed - value mismatch"
        elif isinstance(snmp_value, (Integer, Counter32, Counter64, Gauge32, TimeTicks)):
            if int(set_value) != int(value):
                return False, "Error: Set verification failed - value mismatch"
        elif str(set_value) != str(value):
            return False, "Error: Set verification failed - value mismatch"
            
        return True, "Success"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def determine_snmp_type(oid: str, value: str) -> Any:
    """
    Determine the correct SNMP type for a value based on OID and value content.
    
    Args:
        oid: The SNMP OID
        value: The string value to be converted
        
    Returns:
        SNMP type object with the value
    """
    # OID-specific type mapping
    oid_type_map = {
        # General OIDs
        '1.3.6.1.4.1.60811.1.1.1': OctetString,  # Serial Number
        '1.3.6.1.4.1.60811.1.1.7': IpAddress,    # IP Address
        '1.3.6.1.4.1.60811.1.1.5': OctetString,  # Organization Name
        '1.3.6.1.4.1.60811.1.1.6': OctetString,  # Door Label
        '1.3.6.1.4.1.60811.1.1.9': Integer,      # Silent Flag
        '1.3.6.1.4.1.60811.1.1.27': Integer,     # Triggered Status [Humid.]
        
        # Temperature OIDs
        '1.3.6.1.4.1.60811.1.1.28': Integer,     # High Temp Trigger
        '1.3.6.1.4.1.60811.1.1.29': Integer,     # Low Temp Trigger
        '1.3.6.1.4.1.60811.1.1.32': Integer,     # Unit of Measurement
        '1.3.6.1.4.1.60811.1.1.26': Integer,     # Triggered Status [Temp.]
        
        # Door OIDs
        '1.3.6.1.4.1.60811.1.1.13': Integer,     # Triggered Status
    }
    
    # Get type from map or try to determine from value
    snmp_type = oid_type_map.get(oid)
    if snmp_type:
        if snmp_type == IpAddress:
            try:
                # Validate IP address format
                ipaddress.ip_address(value)
                return IpAddress(value)
            except ValueError:
                raise ValueError(f"Invalid IP address format: {value}")
        elif snmp_type == Integer:
            try:
                # First try direct integer conversion
                return Integer(int(value))
            except ValueError:
                try:
                    # If that fails, try converting from float string
                    return Integer(int(float(value)))
                except ValueError:   
                    raise ValueError(f"Invalid integer value: {value}")
        else:
            return snmp_type(value)
    
    # If no specific mapping, try to determine type from value
    try:
        # Try integer first
        int_val = int(value)
        if 0 <= int_val <= 4294967295:
            return Integer(int_val)
        return Counter64(int_val)
    except ValueError:
        # If not a number, treat as string
        return OctetString(value)


async def handle_device_edit(device: Device):
    """Handle editing device fields via SNMP SET using a dialog"""
    
     # Initialize expander state for this device if not exists
    if device.node_id not in st.session_state.expander_states:
        st.session_state.expander_states[device.node_id] = True
    
    with st.expander(f"Edit Device: {device.name}", expanded=st.session_state.expander_states[device.node_id]):
        menu_categories = {
            "General Information": GENERAL_OIDS,
            "Door": DOOR_OIDS,
            "Temperature": TEMPERATURE_OIDS,
            "Humidity": HUMIDITY_OIDS
        }
        
        category = st.selectbox("Category", list(menu_categories.keys()))
        
        if category:
            oids = menu_categories[category]
            editable_fields = [k for k, v in oids.items() 
                             if k in EDITABLE_FIELDS and EDITABLE_FIELDS[k]]
            
            if editable_fields:
                # Create a container for all fields
                fields_container = st.container()
                
                # Get current values for all fields
                field_values = {}
                for field in editable_fields:
                    oid = oids[field]
                    current_value = await snmp_get(device.ip, oid, device.community)
                    field_values[field] = current_value
                
                # Display all fields with their own input and apply button
                for field in editable_fields:
                    oid = oids[field]
                    with fields_container:
                        st.markdown(f"### {field}")
                        
                        # Handle error cases
                        if field_values[field].startswith("Error:"):
                            st.error(f"Failed to get current value: {field_values[field]}")
                            continue
                            
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            new_value = st.text_input(f"New Value", 
                                                    value=field_values[field],
                                                    key=f"input_{field}")
                        with col2:
                            if st.button("Apply", key=f"apply_{field}"):
                                print(f"Attempting to snmp set for {field}") # Debug
                                success, message = await snmp_set(
                                    device.ip, oid, new_value, device.write_community
                                )
                                
                                if success:
                                    # Calculate new states based on the updated value
                                    print("Calculating device states") # Debug
                                    updates = await calculate_device_states(device, oid, field, category, new_value, st.session_state.snmp_poller)
                                    print(f"Updates calculated: {updates}")  # Debug
                                    # Apply the updates
                                    print("Applying device updates") # Debug
                                    apply_device_updates(device, updates)
                                    # Force a rerun to update the visualization
                                    st.success(f"Successfully updated {field}")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to update {field}: {message}")
                        
                        st.markdown("---")  # Add a separator between fields
                
                # Add a button to close the expander
                if st.button("Close", key=f"close_{device.node_id}"):
                    st.session_state.expander_states[device.node_id] = False
                    st.session_state.editing_device = None
                    st.rerun()

async def scan_network(start_ip: str, end_ip: str, progress_bar) -> List[Device]:
    """Scan IP range for SNMP devices with enhanced detection"""
    devices = []
    try:
        start_int = int(ipaddress.IPv4Address(start_ip))
        end_int = int(ipaddress.IPv4Address(end_ip))
        total_ips = end_int - start_int + 1
        
        for ip_int in range(start_int, end_int + 1):
            current_ip = str(ipaddress.IPv4Address(ip_int))
            progress_bar.progress((ip_int - start_int + 1) / total_ips)
            
            try:
                # First check sysObjectID to identify Door Sentry devices
                sys_object_id = await snmp_get(current_ip, '1.3.6.1.2.1.1.2')
                
                if not isinstance(sys_object_id, str) or sys_object_id.startswith("Error"):
                    continue
                    
                # Verify it's a Door Sentry device by checking if sysObjectID matches our OID
                if '.1.3.6.1.4.1.60811.1.1.1' in sys_object_id:
                    # It's a Door Sentry device - get essential parameters
                    device_serial = await snmp_get(current_ip, '.1.3.6.1.4.1.60811.1.1.1')
                    room_label = await snmp_get(current_ip, '.1.3.6.1.4.1.60811.1.1.6')
                    
                    # Use room label if available, otherwise use IP-based name
                    name = room_label if not isinstance(room_label, str) or not room_label.startswith("Error") else f"Door Sentry at {current_ip}"
                    
                    # Create device with position offset for each new device
                    device = Device(
                        ip=current_ip,
                        name=name,
                        is_snmp=True,
                        community='public',  # Default community
                        write_community='private',  # Default write community
                        version='v2c',
                        position={
                            'x': 100 + len(st.session_state.devices) * 150,
                            'y': 100 + len(st.session_state.devices) * 50
                        }
                    )
                    
                    # Initialize device states in database
                    device_id = device.node_id
                    db.update_device_state(device_id, 'temperature_critical', 'inactive')
                    db.update_device_state(device_id, 'temperature_warning', 'inactive')
                    db.update_device_state(device_id, 'humidity_critical', 'inactive')
                    db.update_device_state(device_id, 'humidity_warning', 'inactive')
                    db.update_device_state(device_id, 'door_open', 'inactive')
                    
                    # Try to get additional parameters if available
                    try:
                        # Temperature parameters
                        temp = await snmp_get(current_ip, '.1.3.6.1.4.1.60811.1.1.54')
                        if not isinstance(temp, str) or not temp.startswith("Error"):
                            pass
                            # put line that saves to device states table in db here
                            
                        # Humidity parameters    
                        humid = await snmp_get(current_ip, '.1.3.6.1.4.1.60811.1.1.55')
                        if not isinstance(humid, str) or not humid.startswith("Error"):
                            pass
                            # put line that saves to device states table in db here
                            
                        # Door status
                        door_status = await snmp_get(current_ip, '.1.3.6.1.4.1.60811.1.1.12')
                        if not isinstance(door_status, str) or not door_status.startswith("Error"):
                            pass
                            # put line that saves to device states table in db here
                            
                    except Exception as e:
                        print(f"Error getting additional parameters for {current_ip}: {str(e)}")
                    
                    # Save device to database and add to list
                    if add_device_to_lists(device):
                        devices.append(device)
                        # Update device status cache
                        cache_key = f"{device.ip}_{device.community}"
                        st.session_state.device_status_cache[cache_key] = True
                        
                        print(f"Found Door Sentry device at {current_ip}")
                        
            except Exception as e:
                print(f"Error scanning {current_ip}: {str(e)}")
                continue
                
    except Exception as e:
        st.error(f"Error during network scan: {str(e)}")
    
    return devices
    

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return ""

def initialize_position_state():
    """Initialize position and viewport tracking in session state"""
    if 'device_positions' not in st.session_state:
        st.session_state.device_positions = {}
    if 'network_viewport' not in st.session_state:
        st.session_state.network_viewport = {
            'scale': 1.0,
            'position': {'x': 0, 'y': 0}
        }
    if 'last_layout' not in st.session_state:
        st.session_state.last_layout = None
    if 'last_viewport_save' not in st.session_state:
        st.session_state.last_viewport_save = 0
    if 'viewport_debounce_ms' not in st.session_state:
        st.session_state.viewport_debounce_ms = 500  # Debounce time in milliseconds

def sync_positions_with_devices():
    """Synchronize stored positions with device objects"""
    print("Syncing positions")  # Debug
    if not hasattr(st.session_state, 'devices'):
        return
        
    device_positions = {}
    for device in st.session_state.devices:
        try:
            # Get position from database with retries
            retries = 3
            stored_position = None
            while retries > 0:
                try:
                    stored_position = db.get_device_position(device.node_id)
                    break
                except Exception as e:
                    print(f"Error getting position (retries left: {retries}): {e}")
                    retries -= 1
                    if retries == 0:
                        raise
            
            if stored_position:
                device.position = stored_position
                device_positions[device.node_id] = stored_position
                print(f"Synced position for {device.name}: {stored_position}")  # Debug
            else:
                # If no position in database, save current position
                print(f"No stored position for {device.name}, saving current position")  # Debug
                db.save_device_position(device.node_id, device.position['x'], device.position['y'])
                device_positions[device.node_id] = device.position
        except Exception as e:
            print(f"Error syncing position for device {device.name}: {e}")


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value between 0 and 1 for SVG positioning"""
    try:
        value = max(min_val, min(max_val, value))
        return (value - min_val) / (max_val - min_val)
    except (ValueError, ZeroDivisionError):
        return 0.5

# First, enhance the devices_to_vis_elements function to include state information
def devices_to_vis_elements(devices: List[Device]) -> Tuple[List[Dict], List[Dict]]:
    """Convert devices to Vis.js compatible format with state-based styling"""
    print("\nGenerating vis elements")  # Debug
    nodes = []
    edges = []

    for device in st.session_state.devices:
        #print(f"\nProcessing device: {device.name}")  # Debug
        
        # Get device states
        device_states = db.get_all_device_states(device.node_id)
        #print(f"Device states from DB: {device_states}")  # Debug
        
        # Use state values instead of querying again
        current_temp = float(device_states.get('temperature_value', '25.0'))
        current_humid = float(device_states.get('humidity_value', '50.0'))
        door_status = device_states.get('door_status', 'CLOSED')
        
        #print(f"Using values - Temp: {current_temp}, Humid: {current_humid}, Door: {door_status}")  # Debug
        
        
        # Use device states to build node state object
        node_state = {
            'temperature': float(device_states.get('temperature_value', 25.0)),
            'humidity': float(device_states.get('humidity_value', 50.0)),
            'doorStatus': device_states.get('door_status', 'CLOSED'),
            'tempWarningHigh': float(device_states.get('temperature_high_warning', 35.0)),
            'tempWarningLow': float(device_states.get('temperature_low_warning', 15.0)),
            'tempCriticalHigh': float(device_states.get('temperature_high_critical', 40.0)),
            'tempCriticalLow': float(device_states.get('temperature_low_critical', 10.0)),
            'humidityWarningHigh': float(device_states.get('humidity_high_warning', 70.0)),
            'humidityWarningLow': float(device_states.get('humidity_low_warning', 30.0)),
            'humidityCriticalHigh': float(device_states.get('humidity_high_critical', 80.0)),
            'humidityCriticalLow': float(device_states.get('humidity_low_critical', 20.0)),
            'temperatureState': device_states.get('temperature_state', 'normal'),
            'humidityState': device_states.get('humidity_state', 'normal')
        }
        #print(f"Node state object: {node_state}")  # Debug

        node = {
            'id': device.node_id,
            'label': f"{device.name}\n({device.ip})",
            'x': device.position['x'],
            'y': device.position['y'],
            'shape': 'image',
            'state': node_state,
            'size': 60
        }
        
        #print(f"Created node: {node}")  # Debug
        
        # Apply states to node style
        node_style = get_node_style(device_states)
        node.update(node_style)
        
        # Update label with current values
        status_lines = [
            f"Temp: {current_temp}°C",
            f"Humidity: {current_humid}%"
        ]
        
        if door_status == 'OPEN':
            status_lines.append("Door: OPEN")
        
        node['label'] = node['label'] + '\n' + '\n'.join(status_lines)
        nodes.append(node)
    
    return nodes, edges

def get_node_style(device_states: Dict[str, str]) -> Dict:
    """Get node visual style based on device states with improved selection highlights"""
    style = {
        'color': {
            'background': '#97C2FC',  # Default background
            'border': '#2B7CE9',      # Default border
            'highlight': {
                'background': '#D2E5FF',  # Default highlight
                'border': '#2B7CE9'
            }
        },
        'borderWidth': 2,
        'shape': 'box'
    }
    
    # Get temperature and humidity states
    temp_state = device_states.get('temperature_state', 'normal')
    humid_state = device_states.get('humidity_state', 'normal')
    
    # Priority-based state coloring with matching highlights
    if temp_state == 'critical_high' or device_states.get('temperature_critical_high', 'inactive') == 'active':
        style['color'].update({
            'background': '#FF4444',  # Bright red for high temperature
            'border': '#CC0000',
            'highlight': {
                'background': '#FF6666',  # Lighter red for highlight
                'border': '#CC0000'
            }
        })
        style['borderWidth'] = 3
    elif temp_state == 'critical_low' or device_states.get('temperature_critical_low', 'inactive') == 'active':
        style['color'].update({
            'background': '#4444FF',  # Bright blue for low temperature
            'border': '#0000CC',
            'highlight': {
                'background': '#6666FF',  # Lighter blue for highlight
                'border': '#0000CC'
            }
        })
        style['borderWidth'] = 3
    elif humid_state == 'critical_high' or device_states.get('humidity_critical_high', 'inactive') == 'active':
        style['color'].update({
            'background': '#8B008B',  # Dark magenta for high humidity
            'border': '#4B0082',
            'highlight': {
                'background': '#9B209B',  # Lighter magenta for highlight
                'border': '#4B0082'
            }
        })
        style['borderWidth'] = 3
    elif humid_state == 'critical_low' or device_states.get('humidity_critical_low', 'inactive') == 'active':
        style['color'].update({
            'background': '#20B2AA',  # Light sea green for low humidity
            'border': '#008B8B',
            'highlight': {
                'background': '#40D2CA',  # Lighter sea green for highlight
                'border': '#008B8B'
            }
        })
        style['borderWidth'] = 3
    elif temp_state == 'warning_high' or device_states.get('temperature_warning_high', 'inactive') == 'active':
        style['color'].update({
            'background': '#FFA500',  # Orange for high temperature warning
            'border': '#CC8400',
            'highlight': {
                'background': '#FFB52E',  # Lighter orange for highlight
                'border': '#CC8400'
            }
        })
    elif temp_state == 'warning_low' or device_states.get('temperature_warning_low', 'inactive') == 'active':
        style['color'].update({
            'background': '#87CEEB',  # Light blue for low temperature warning
            'border': '#4682B4',
            'highlight': {
                'background': '#A7EEFB',  # Lighter sky blue for highlight
                'border': '#4682B4'
            }
        })
    elif humid_state == 'warning_high' or device_states.get('humidity_warning_high', 'inactive') == 'active':
        style['color'].update({
            'background': '#DDA0DD',  # Plum for high humidity warning
            'border': '#DA70D6',
            'highlight': {
                'background': '#EEBFEE',  # Lighter plum for highlight
                'border': '#DA70D6'
            }
        })
    elif humid_state == 'warning_low' or device_states.get('humidity_warning_low', 'inactive') == 'active':
        style['color'].update({
            'background': '#98FB98',  # Pale green for low humidity warning
            'border': '#90EE90',
            'highlight': {
                'background': '#B8FBB8',  # Lighter pale green for highlight
                'border': '#90EE90'
            }
        })
    elif device_states.get('door_open', 'inactive') == 'active' or device_states.get('door_status', 'CLOSED') == 'OPEN':
        style['color'].update({
            'background': '#FFFF00',  # Yellow for open door
            'border': '#CCCC00',
            'highlight': {
                'background': '#FFFF66',  # Lighter yellow for highlight
                'border': '#CCCC00'
            }
        })
    
    return style


# Update create_vis_options to include state-based styling
def create_vis_options() -> Dict:
    """Create Vis.js network options with state handling"""
    return {
        "nodes": {
            "shape": "box",
            "size": 30,
            "font": {
                "size": 14,
                "multi": True,
                "face": "arial",
                "align": "center"
            },
            "borderWidth": 2,
            "shadow": True,
            "fixed": {
                "x": False,
                "y": False
            },
            "physics": False,  # Disable physics for individual nodes
            "color": {
                "border": "#2B7CE9",
                "background": "#97C2FC",
                "highlight": {
                    "border": "#2B7CE9",
                    "background": "#D2E5FF"
                }
            }
        },
        "groups": {
            "normal": {
                "color": {
                    "background": "#97C2FC",
                    "border": "#2B7CE9"
                }
            },
            "warning": {
                "color": {
                    "background": "#FFA500",
                    "border": "#CC8400"
                }
            },
            "critical": {
                "color": {
                    "background": "#FF4444",
                    "border": "#CC0000"
                }
            },
            "alert": {
                "color": {
                    "background": "#FFFF00",
                    "border": "#CCCC00"
                }
            }
        },
        "physics": {
            "enabled": True,
            "stabilization": {
                "enabled": True,
                "iterations": 100,  # Reduced from default
                "updateInterval": 50,  # More frequent updates during stabilization
                "fit": False  # Don't fit to view after stabilization
            }
        },
        
        "interaction": {
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "selectable": True,
            "multiselect": True,
            "hover": True,
            "navigationButtons": False,  # Disable navigation buttons to reduce UI complexity
            "keyboard": {
                "enabled": True,
                "bindToWindow": False  # Only enable keyboard when network is focused
            },
            "hideEdgesOnDrag": True,  # Hide edges while dragging to improve performance
            "hideNodesOnDrag": False  # Keep nodes visible during drag
        }
    }

def create_topology_stylesheet():
    """Create stylesheet with icon support"""
    return [
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'background-color': '#6495ED',
                'width': 100,
                'height': 60,
                'font-size': 12,
                'text-wrap': 'wrap',
                'background-image': 'data(backgroundImage)',
                'background-fit': 'contain',
                'background-opacity': 'data(backgroundOpacity)',
                'background-width': 'data(backgroundWidth)',
                'background-height': 'data(backgroundHeight)'
            }
        },
        {
            'selector': '.with-icon',
            'style': {
                'text-margin-y': 20,
                'background-color': 'data(backgroundColor)'
            }
        },
        {
            'selector': '.snmp',
            'style': {
                'background-color': '#6495ED'
            }
        },
        {
            'selector': '.non-snmp',
            'style': {
                'background-color': '#98FB98'
            }
        },
        {
            'selector': ':selected',
            'style': {
                'border-color': '#FFD700',
                'border-width': 3
            }
        }
    ]

def handle_shutdown():
    """Handle graceful shutdown of async tasks"""
    for task in asyncio.all_tasks():
        task.cancel()

def initialize_app_state():
    """Initialize app state from database"""
    print("Initializing app state") # Debug
    
    # Clean up database first
    db.clean_database()
    
    # Load devices from database
    devices_data = db.get_all_devices()
    st.session_state.devices = []
    
    for device_data in devices_data:
        device = Device.from_dict(device_data)
        if device.position:
            print(f"Loaded device {device.name} with position {device.position}")
        st.session_state.devices.append(device)
        print(f"Loaded device {device.name} with position {device.position}")  # Debug
    
    # Initialize device status cache
    if 'device_status_cache' not in st.session_state:
        st.session_state.device_status_cache = {}
    
    # Load background
    bg_data = db.get_background_image()
    if bg_data:
        st.session_state.background_image = bg_data['image']
        st.session_state.background_dimensions = bg_data['dimensions']

def create_topology_view():
    """Create and display the network topology view"""

    if 'network_initialized' not in st.session_state:
        st.session_state.network_initialized = False

    if 'query_results' not in st.session_state:
        st.session_state.query_results = {}
    if 'menu_expanded' not in st.session_state:
        st.session_state.menu_expanded = {}


    if 'current_category' not in st.session_state:
        st.session_state.current_category = {}


    try:
        #st.title("Network Topology")
        
        # Initialize States
        initialize_position_state()

        # Call this at the start of create_topology_view()
        if 'app_initialized' not in st.session_state:
            initialize_app_state()
            st.session_state.app_initialized = True
        
        if 'reload_pending' not in st.session_state:
            st.session_state.reload_pending = False

        if 'devices' not in st.session_state:
            st.session_state.devices = []
            
        if 'last_component_value' not in st.session_state:
            st.session_state.last_component_value = None
            
        if 'editing_device' not in st.session_state:
            st.session_state.editing_device = None
            
        viewport_state = db.get_viewport_state()
        if viewport_state is None:
            viewport_state = {
                'scale': 1.0,
                'position': {'x': 0, 'y': 0}
            }
        else:
            print(f"Former viewport loaded: {viewport_state}")
        
        # Device Management in Sidebar
        with st.sidebar:
            st.markdown("---")
            st.subheader("Add Devices")
            
            # Add Device Form
            with st.form("add_device_form_topology"):
                name = st.text_input("Device Name", key="name_input_topology")
                ip = st.text_input("IP Address", key="ip_input_topology")
                is_snmp = st.checkbox("SNMP Device", value=True, key="snmp_check_topology")
                simulation_mode = st.checkbox("Simulation Mode", value=False, help="Generate simulated sensor readings for testing")
                
                # SNMP settings in collapsible section
                if is_snmp:
                    with st.expander("SNMP Settings", expanded=True):
                        community = st.text_input("Read Community", value="public", key="community_input_topology")
                        write_community = st.text_input("Write Community", value="private", key="write_community_input_topology")
                        version = st.selectbox("SNMP Version", ["v1", "v2c"], key="version_select_topology")
                else:
                    community = "public"
                    write_community = "private"
                    version = "v2c"
                    
                if st.form_submit_button("Add Device"):
                    if name and ip:
                        try:
                            ipaddress.ip_address(ip)
                            if any(d.ip == ip for d in st.session_state.devices):
                                st.error("Device with this IP already exists!")
                            else:
                                new_device = Device(
                                    ip=ip,
                                    name=name,
                                    is_snmp=is_snmp,
                                    community=community,
                                    write_community=write_community,
                                    version=version,
                                    position={'x': 100 + len(st.session_state.devices) * 100, 
                                            'y': 100 + len(st.session_state.devices) * 50},
                                    simulation_mode = simulation_mode
                                )
                                
                                # Save to database with enhanced data
                                db.save_device({
                                    'node_id': new_device.node_id,
                                    'name': new_device.name,
                                    'ip': new_device.ip,
                                    'is_snmp': new_device.is_snmp,
                                    'community': new_device.community,
                                    'write_community': new_device.write_community,
                                    'version': new_device.version,
                                    'position': new_device.position, 
                                    'simulation_mode': new_device.simulation_mode
                                })
                                
                                # Initialize device states in database
                                db.update_device_state(new_device.node_id, 'temperature_critical', 'inactive')
                                db.update_device_state(new_device.node_id, 'temperature_warning', 'inactive')
                                db.update_device_state(new_device.node_id, 'door_open', 'inactive')
                                
                                st.session_state.devices.append(new_device)
                                st.success(f"Device {name} ({ip}) added successfully!")
                                st.rerun()
                        except ValueError:
                            st.error("Invalid IP address format")
            
            # Device List and Remove Option
            if st.session_state.devices:
                st.markdown("---")
                st.subheader("Remove Devices")
                
                device_options = {f"{d.name} ({d.ip})": idx for idx, d in enumerate(st.session_state.devices)}
                
                selected_device = st.selectbox(
                    "Select Device",
                    options=list(device_options.keys()),
                    key="remove_device_select_topology"
                )

                # Add confirmation state
                if 'confirm_remove_topology' not in st.session_state:
                    st.session_state.confirm_remove_topology = False
                
                if selected_device:
                    col1, col2 = st.columns([3,1])
                    with col1:
                        st.session_state.confirm_remove_topology = st.checkbox(
                            "Confirm removal",
                            key="confirm_remove_topology_checkbox"
                        )
                    with col2:
                        if st.button("Remove", key="remove_device_button_topology"):
                            if st.session_state.confirm_remove_topology:
                                try:
                                    idx = device_options[selected_device]
                                    device = st.session_state.devices[idx]
                                    
                                    # Remove from database first
                                    db.remove_device(device.node_id)
                                    
                                    # Then remove from session state
                                    st.session_state.devices.pop(idx)
                                    
                                    # Clean up associated data
                                    if device.node_id in st.session_state.device_status_history:
                                        del st.session_state.device_status_history[device.node_id]
                                    if device.node_id in st.session_state.last_update_times:
                                        del st.session_state.last_update_times[device.node_id]
                                    
                                    # Clear cache for removed device
                                    cache_key = f"{device.ip}_{device.community}"
                                    if cache_key in st.session_state.device_status_cache:
                                        del st.session_state.device_status_cache[cache_key]
                                    
                                    st.success(f"Device {device.name} removed successfully!")
                                    st.session_state.confirm_remove_topology = False
                                    time.sleep(0.5)  # Brief pause to show success message
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error removing device: {str(e)}")
                            else:
                                st.warning("Please confirm removal by checking the box")
        add_polling_controls()

        # Sync positions before creating elements
        sync_positions_with_devices()
        
        # Use stored viewport state if it exists, otherwise use defaults
        #viewport_state = st.session_state.network_viewport if st.session_state.get('network_viewport') else {
        #    'scale': 1.0,
        #    'position': {'x': 0, 'y': 0}
        #}

        print("Setting up network component...")  # Debug
        # Always generate network elements with current state
        nodes, edges = devices_to_vis_elements(st.session_state.devices)
        st.session_state.network_elements = {
            'nodes': nodes,
            'edges': edges,
            'key': str(time.time())
        }

        current_key = st.session_state.network_elements.get('key', 'default')
        #print(f"Using network elements key: {current_key}")  # Debug
            
        options = create_vis_options()
        
        # Before creating the vis_network component:
        background_image = st.session_state.get('background_image')
        
        # For debugging, let's print the actual props we're sending
        debug_props = {
            'nodes_count': len(nodes),
            'edges_count': len(edges),
            'has_background': bool(background_image),
            'background_size': len(background_image) if background_image else 0,
            'has_viewport': bool(viewport_state),
            'viewport_details': viewport_state,
            'network_initialized': st.session_state.get('network_initialized', False),
            'current_scale': viewport_state.get('scale', 1.0),
            'current_position': viewport_state.get('position', {'x': 0, 'y': 0})
        }
        #st.write("Debug - Component props:", debug_props)
        
        # Use a data URL directly for testing
        test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAAKklEQVQ4jWNk+M9Qz0BFwMRAZTBq4KiBowaOGjhq4KiB5AcGIyM1SxoGBgYAoL4GVQ6rduEAAAAASUVORK5CYII="

        # Before creating vis_network component:
        current_elements = st.session_state.get('network_elements', {})
        #print(f"Current network elements key: {current_elements.get('key')}")  # Debug

        # Create network component with key-based updates
        selected = vis_network(
            nodes=st.session_state.network_elements['nodes'],
            edges=st.session_state.network_elements['edges'],
            options=options,
            background_image=background_image if background_image else test_image,
            viewport=viewport_state,
            #key=f"topology_{current_key}",
            key="topology_network",
            height="600px"
        )
        
        #print(f"Raw selected value: {selected}")  # Debug raw value
        
        if selected != st.session_state.last_component_value:
            st.session_state.last_component_value = selected
        
        # After successful network creation
        st.session_state.network_initialized = True

        # Handle selection and viewport updates
        if selected and isinstance(selected, dict):
            print(f"Selected dict keys: {selected.keys()}")  # Debug keys
            
            update_type = selected.get('type')
            print(f"Update Type: {update_type}")
            
            # Handle viewport updates with debouncing
            if update_type == 'viewport':
                
                viewport = selected.get('viewport')
                
                if viewport:
                    #print(f"Saving viewport state: scale={viewport['scale']}, position={viewport['position']}")
                    current_time = time.time() * 1000  # Convert to milliseconds
                    time_since_last_save = current_time - st.session_state.last_viewport_save
                    
                    # Only save if enough time has passed since last save
                    if time_since_last_save >= st.session_state.viewport_debounce_ms:
                        db.save_viewport_state(viewport['scale'], viewport['position'])
                        st.session_state.last_viewport_save = current_time
                    
                    # Always update the current viewport state in memory
                    viewport_state = viewport
                    st.session_state.network_viewport = viewport_state
                    #print(f"Viewport updated: {viewport_state}")  # Debug
                    
                    # Restore expander state
                    if st.session_state.selected_device_id:
                        st.session_state.expander_states[st.session_state.selected_device_id] = True
                    
            elif update_type == 'context_menu':
                node_id = selected.get("nodeId")
                if node_id:
                    device = next((d for d in st.session_state.devices 
                                if d.node_id == node_id), None)
                    if device:
                        # Create a new asyncio event loop to handle the async operation
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(handle_device_edit(device))
                        finally:
                            loop.close()
                        
            # Handle position updates with improved efficiency
            elif update_type == 'position':
                positions_updated = False
                try:
                    for node_id, position in selected['positions'].items():
                        #print(f"Updating position for {node_id}: {position}")  # Debug
                        
                        # Save to database with retries
                        retries = 3
                        while retries > 0:
                            try:
                                # Save to database
                                db.save_device_position(node_id, position['x'], position['y'])
                                break
                            except Exception as e:
                                print(f"Error saving position (retries left: {retries}): {e}")
                                retries -= 1
                                if retries == 0:
                                    raise
                        
                        # Update device object
                        device = next(
                            (d for d in st.session_state.devices if d.node_id == node_id),
                            None
                        )
                        if device:
                            device.position = position
                            positions_updated = True
                           #print(f"Updated device {device.name} position to {position}")  # Debug
                except Exception as e:
                    print(f"Error updating positions: {e}")
                    
                if positions_updated:
                    # Force position sync after updates
                    sync_positions_with_devices()

                        
            # Selection Handling
            elif update_type == 'selection':
                node_id = selected['selected'][0]
                device = next(
                    (d for d in st.session_state.devices if d.node_id == node_id),
                    None
                )
                
                if device:
                    st.session_state.selected_device = device
                    st.session_state.selected_device_id = device.node_id
                    
                    # initialize expander state
                    if device.node_id not in st.session_state.expander_states:
                        st.session_state.expander_states[device.node_id] = True
                    
                    # Device Details Section
                    with st.sidebar:
                        st.header("Device Details")
                        
                        # Basic Info
                        st.markdown(f"**Name:** {device.name}")
                        st.markdown(f"**IP:** {device.ip}")
                        st.markdown(f"**Type:** {'SNMP' if device.is_snmp else 'Non-SNMP'}")
                        
                        if device.is_snmp:
                            st.markdown(f"**Community:** {device.community}")
                            st.markdown(f"**Version:** {device.version}")
                                            
            # Clear selection if no device is selected
            #elif not selected.get('selected'):
            #    st.session_state.selected_device_id = None

        # Layout Controls
        with st.sidebar:
            st.markdown("---")
            st.subheader("Layout Controls")
            
            # Display Options
            with st.expander("Display Options", expanded=False):
                st.checkbox("Show Labels", value=True, key="show_labels")
                st.checkbox("Show IP Addresses", value=True, key="show_ips")
                st.checkbox("Show Status Indicators", value=True, key="show_status")
                st.slider("Node Size", min_value=50, max_value=150, value=100, key="node_size")

        with st.sidebar:
            st.markdown("---")
            st.header("Background Image")
            
            with st.form("background_image_form"):
                uploaded_bg = st.file_uploader(
                    "Upload background image",
                    type=['png', 'jpg', 'jpeg'],
                    key="bg_uploader"
                )
                
                # Submit button for the form
                submit_image = st.form_submit_button("Upload Background")
                
                if submit_image and uploaded_bg is not None:
                    try:
                        image = Image.open(uploaded_bg)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Convert to base64
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        img_data_url = f"data:image/png;base64,{img_str}"
                        
                        dimensions = {
                            'width': image.width,
                            'height': image.height
                        }
                        
                        # Save to database
                        db.save_background_image(img_data_url, dimensions)
                        
                        # Update session state
                        st.session_state.background_image = img_data_url
                        st.session_state.background_dimensions = dimensions
                        
                        st.success("Background image uploaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error uploading background image: {str(e)}")
                        st.exception(e)
            
            # Clear background - outside the form
            if st.session_state.get('background_image') and st.button("Clear Background"):
                st.session_state.background_image = None
                st.session_state.background_dimensions = {'width': 1280, 'height': 1000}
                db.save_background_image(None, {'width': 1280, 'height': 1000})
                st.rerun()
        
        with st.sidebar:
            st.markdown("---")
            st.subheader("Topology Management")
            
            # Save current topology
            save_topology()
            
            # Load topology
            uploaded_file = st.file_uploader("Load Topology", type="json", key="topology_upload")
            if uploaded_file is not None and not st.session_state.reload_pending:
                st.session_state.reload_pending = True
                if load_topology(uploaded_file):
                    st.rerun()
                st.session_state.reload_pending = False
            
            # Clear Topology
            if st.button("Clear Topology"):
                if hasattr(st.session_state, 'devices') and st.session_state.devices:
                    if st.checkbox("Confirm Clear All Data"):
                        # Clear database
                        db.clear_all_data()
                        
                        # Clear session state
                        st.session_state.devices = []
                        st.session_state.background_image = None
                        st.session_state.background_dimensions = {'width': 1280, 'height': 1000}
                        
                        st.rerun()

        # Monitor Status with proper error handling
        if hasattr(st.session_state, 'devices') and st.session_state.devices:
            with st.sidebar:
                st.markdown("---")
                st.subheader("Status Monitor")
                
                monitor_container = st.empty()
                with monitor_container:
                    status_table = []
                    for device in st.session_state.devices:
                        status = "PLACEHOLDER (get cached device status)"
                        last_update = "PLACEHOLDER (get last status check time)"
                        status_table.append({
                            "Device": f"{device.name} ({device.ip})",
                            "Status": "Active" if status else "Inactive",
                            "Last Update": last_update
                        })
                    
                    st.table(status_table)
                    
                    if st.button("Refresh Status"):
                        st.rerun()
            
    except Exception as e:
        st.error(f"Error in topology view: {str(e)}")


async def calculate_device_states(device: Device, oid: str, field_name: str, category: str, value: str, poller: SNMPPoller) -> Dict:
    """Calculate states without database operations"""
    try:
        updates = {
            'value': value,
            'error': False,
            'snmp_updates': [],
            'state_updates': []
        }
        
        # Process the value based on category and field
        if category == "Temperature" and field_name == "Live Temperature":
            try:
                current_temp = float(value.rstrip('CF'))
                
                # Add temperature value to state updates
                updates['state_updates'].extend([
                    ('temperature_value', str(current_temp)),
                    ('temperature', str(current_temp))
                ])
                
                # Get threshold values from cache
                high_temp_result = await poller.get_cached_threshold(
                    device, 
                    TEMPERATURE_OIDS['High Temp. Trigger Set Point'],
                    'Temperature',
                    'High Temp. Trigger Set Point'
                )
                low_temp_result = await poller.get_cached_threshold(
                    device,
                    TEMPERATURE_OIDS['Low Temp. Trigger Set Point'],
                    'Temperature',
                    'Low Temp. Trigger Set Point'
                )
                
                # Add original value and thresholds to snmp_updates
                updates['snmp_updates'].extend([
                    {
                        'oid': oid,
                        'value': str(current_temp),
                        'category': category,
                        'field_name': field_name
                    }
                ])
                
                # Only add threshold updates if they were freshly fetched (not from cache)
                if (device.node_id, TEMPERATURE_OIDS['High Temp. Trigger Set Point']) not in poller.threshold_cache:
                    updates['snmp_updates'].append({
                        'oid': TEMPERATURE_OIDS['High Temp. Trigger Set Point'],
                        'value': high_temp_result,
                        'category': 'Temperature',
                        'field_name': 'High Temp. Trigger Set Point'
                    })
                if (device.node_id, TEMPERATURE_OIDS['Low Temp. Trigger Set Point']) not in poller.threshold_cache:
                    updates['snmp_updates'].append({
                        'oid': TEMPERATURE_OIDS['Low Temp. Trigger Set Point'],
                        'value': low_temp_result,
                        'category': 'Temperature',
                        'field_name': 'Low Temp. Trigger Set Point'
                    })
                
                # Get numerical values for comparison
                high_temp = float(high_temp_result) if not high_temp_result.startswith("Error") else 35.0
                low_temp = float(low_temp_result) if not low_temp_result.startswith("Error") else 10.0
                
                # Calculate states
                if current_temp >= high_temp:
                    updates['state_updates'].extend([
                        ('temperature_state', 'critical_high'),
                        ('temperature_critical_high', 'active'),
                        ('temperature_warning_high', 'inactive'),
                        ('temperature_critical_low', 'inactive'),
                        ('temperature_warning_low', 'inactive'),
                        ('temperature_critical', 'active'),
                        ('temperature_warning', 'inactive')
                    ])
                elif current_temp >= (high_temp - 5):
                    updates['state_updates'].extend([
                        ('temperature_state', 'warning_high'),
                        ('temperature_critical_high', 'inactive'),
                        ('temperature_warning_high', 'active'),
                        ('temperature_critical_low', 'inactive'),
                        ('temperature_warning_low', 'inactive'),
                        ('temperature_critical', 'inactive'),
                        ('temperature_warning', 'active')
                    ])
                elif current_temp <= low_temp:
                    updates['state_updates'].extend([
                        ('temperature_state', 'critical_low'),
                        ('temperature_critical_high', 'inactive'),
                        ('temperature_warning_high', 'inactive'),
                        ('temperature_critical_low', 'active'),
                        ('temperature_warning_low', 'inactive'),
                        ('temperature_critical', 'active'),
                        ('temperature_warning', 'inactive')
                    ])
                elif current_temp <= (low_temp + 5):
                    updates['state_updates'].extend([
                        ('temperature_state', 'warning_low'),
                        ('temperature_critical_high', 'inactive'),
                        ('temperature_warning_high', 'inactive'),
                        ('temperature_critical_low', 'inactive'),
                        ('temperature_warning_low', 'active'),
                        ('temperature_critical', 'inactive'),
                        ('temperature_warning', 'active')
                    ])
                else:
                    updates['state_updates'].extend([
                        ('temperature_state', 'normal'),
                        ('temperature_critical_high', 'inactive'),
                        ('temperature_warning_high', 'inactive'),
                        ('temperature_critical_low', 'inactive'),
                        ('temperature_warning_low', 'inactive'),
                        ('temperature_critical', 'inactive'),
                        ('temperature_warning', 'inactive')
                    ])
                    
            except ValueError as e:
                print(f"Error processing temperature: {str(e)}")
                updates['error'] = True
                
        elif category == "Humidity" and field_name == "Live Humidity":
            try:
                current_humid = float(value.replace('% RH', '').strip())
                
                # Add humidity value to state updates
                updates['state_updates'].extend([
                    ('humidity_value', str(current_humid)),
                    ('humidity', str(current_humid))
                ])
                
                # Get threshold values from cache
                high_humid_result = await poller.get_cached_threshold(
                    device,
                    HUMIDITY_OIDS['High Humid. Trigger Set Point'],
                    'Humidity',
                    'High Humid. Trigger Set Point'
                )
                low_humid_result = await poller.get_cached_threshold(
                    device,
                    HUMIDITY_OIDS['Low Humid. Trigger Set Point'],
                    'Humidity',
                    'Low Humid. Trigger Set Point'
                )
                
                # Add original value and thresholds to snmp_updates
                updates['snmp_updates'].extend([
                    {
                        'oid': oid,
                        'value': str(current_humid),
                        'category': category,
                        'field_name': field_name
                    }
                ])
                
                # Only add threshold updates if they were freshly fetched (not from cache)
                if (device.node_id, HUMIDITY_OIDS['High Humid. Trigger Set Point']) not in poller.threshold_cache:
                    updates['snmp_updates'].append({
                        'oid': HUMIDITY_OIDS['High Humid. Trigger Set Point'],
                        'value': high_humid_result,
                        'category': 'Humidity',
                        'field_name': 'High Humid. Trigger Set Point'
                    })
                if (device.node_id, HUMIDITY_OIDS['Low Humid. Trigger Set Point']) not in poller.threshold_cache:
                    updates['snmp_updates'].append({
                        'oid': HUMIDITY_OIDS['Low Humid. Trigger Set Point'],
                        'value': low_humid_result,
                        'category': 'Humidity',
                        'field_name': 'Low Humid. Trigger Set Point'
                    })
                
                # Get numerical values for comparison
                high_humid = float(high_humid_result) if not high_humid_result.startswith("Error") else 80.0
                low_humid = float(low_humid_result) if not low_humid_result.startswith("Error") else 20.0
                
                # Calculate states
                if current_humid >= high_humid:
                    updates['state_updates'].extend([
                        ('humidity_state', 'critical_high'),
                        ('humidity_critical_high', 'active'),
                        ('humidity_warning_high', 'inactive'),
                        ('humidity_critical_low', 'inactive'),
                        ('humidity_warning_low', 'inactive'),
                        ('humidity_critical', 'active'),
                        ('humidity_warning', 'inactive')
                    ])
                elif current_humid >= (high_humid - 10):
                    updates['state_updates'].extend([
                        ('humidity_state', 'warning_high'),
                        ('humidity_critical_high', 'inactive'),
                        ('humidity_warning_high', 'active'),
                        ('humidity_critical_low', 'inactive'),
                        ('humidity_warning_low', 'inactive'),
                        ('humidity_critical', 'inactive'),
                        ('humidity_warning', 'active')
                    ])
                elif current_humid <= low_humid:
                    updates['state_updates'].extend([
                        ('humidity_state', 'critical_low'),
                        ('humidity_critical_high', 'inactive'),
                        ('humidity_warning_high', 'inactive'),
                        ('humidity_critical_low', 'active'),
                        ('humidity_warning_low', 'inactive'),
                        ('humidity_critical', 'active'),
                        ('humidity_warning', 'inactive')
                    ])
                elif current_humid <= (low_humid + 10):
                    updates['state_updates'].extend([
                        ('humidity_state', 'warning_low'),
                        ('humidity_critical_high', 'inactive'),
                        ('humidity_warning_high', 'inactive'),
                        ('humidity_critical_low', 'inactive'),
                        ('humidity_warning_low', 'active'),
                        ('humidity_critical', 'inactive'),
                        ('humidity_warning', 'active')
                    ])
                else:
                    updates['state_updates'].extend([
                        ('humidity_state', 'normal'),
                        ('humidity_critical_high', 'inactive'),
                        ('humidity_warning_high', 'inactive'),
                        ('humidity_critical_low', 'inactive'),
                        ('humidity_warning_low', 'inactive'),
                        ('humidity_critical', 'inactive'),
                        ('humidity_warning', 'inactive')
                    ])
                    
            except ValueError as e:
                print(f"Error processing humidity: {str(e)}")
                updates['error'] = True
                
        elif category == "Door" and field_name == "Triggered Status":
            updates['snmp_updates'].append({
                'oid': oid,
                'value': int(float(value)),
                'category': category,
                'field_name': field_name
            })
            
            door_state = 'OPEN' if int(float(value)) == 1 else 'CLOSED'
            updates['state_updates'].extend([
                ('door_status', door_state),
                ('door_open', 'active' if int(float(value)) == 1 else 'inactive'),
                ('door_value', int(float(value)))
            ])
        
        elif category == "Temperature":
            if field_name == "High Temp. Trigger Set Point":
                try:
                    high_temp = float(value)
                    # Add to snmp_updates
                    updates['snmp_updates'].append({
                        'oid': oid,
                        'value': str(high_temp),
                        'category': category,
                        'field_name': field_name
                    })
                    # Add to state_updates
                    updates['state_updates'].extend([
                        ('temperature_high_critical', str(high_temp)),
                        ('temperature_high_warning', str(high_temp - 5))
                    ])
                    #print(f"Added high temp updates: {updates}")  # Debug
                    
                    # Get current temperature for comparison
                    try:
                        current_temp_result = await snmp_get(device.ip, TEMPERATURE_OIDS['Live Temperature'], device.community)
                        current_temp = float(current_temp_result.rstrip('CF')) if not current_temp_result.startswith("Error") else None
                    except Exception as e:
                        print(f"Error getting current temperature: {str(e)}")
                        current_temp = None
                    
                    if current_temp is not None:
                        # Calculate states based on new high trigger and current temp
                        if current_temp >= high_temp:
                            updates['state_updates'].extend([
                                ('temperature_state', 'critical_high'),
                                ('temperature_critical_high', 'active'),
                                ('temperature_warning_high', 'inactive'),
                                ('temperature_critical_low', 'inactive'),
                                ('temperature_warning_low', 'inactive'),
                                ('temperature_critical', 'active'),
                                ('temperature_warning', 'inactive')
                            ])
                        elif current_temp >= (high_temp - 5):
                            updates['state_updates'].extend([
                                ('temperature_state', 'warning_high'),
                                ('temperature_critical_high', 'inactive'),
                                ('temperature_warning_high', 'active'),
                                ('temperature_critical_low', 'inactive'),
                                ('temperature_warning_low', 'inactive'),
                                ('temperature_critical', 'inactive'),
                                ('temperature_warning', 'active')
                            ])
                    
                    updates['snmp_updates'].append({
                        'oid': oid,
                        'value': str(high_temp),
                        'category': category,
                        'field_name': field_name
                    })
                except ValueError as e:
                    print(f"Error processing high temperature trigger: {str(e)}")
                    updates['error'] = True
            
            elif field_name == "Low Temp. Trigger Set Point":
                try:
                    low_temp = float(value)
                    # Add to snmp_updates
                    updates['snmp_updates'].append({
                        'oid': oid,
                        'value': str(low_temp),
                        'category': category,
                        'field_name': field_name
                    })
                    # Update the critical low trigger point state
                    updates['state_updates'].extend([
                        ('temperature_low_critical', str(low_temp)),  # This matches devices_to_vis_elements
                        ('temperature_low_warning', str(low_temp + 5))  # For warning threshold
                    ])

                    # Get current temperature for comparison
                    try:
                        current_temp_result = await snmp_get(device.ip, TEMPERATURE_OIDS['Live Temperature'], device.community)
                        current_temp = float(current_temp_result.rstrip('CF')) if not current_temp_result.startswith("Error") else None
                    except Exception as e:
                        print(f"Error getting current temperature: {str(e)}")
                        current_temp = None
                    
                    if current_temp is not None:
                        # Calculate states based on new low trigger and current temp
                        if current_temp <= low_temp:
                            updates['state_updates'].extend([
                                ('temperature_state', 'critical_low'),
                                ('temperature_critical_high', 'inactive'),
                                ('temperature_warning_high', 'inactive'),
                                ('temperature_critical_low', 'active'),
                                ('temperature_warning_low', 'inactive'),
                                ('temperature_critical', 'active'),
                                ('temperature_warning', 'inactive')
                            ])
                        elif current_temp <= (low_temp + 5):
                            updates['state_updates'].extend([
                                ('temperature_state', 'warning_low'),
                                ('temperature_critical_high', 'inactive'),
                                ('temperature_warning_high', 'inactive'),
                                ('temperature_critical_low', 'inactive'),
                                ('temperature_warning_low', 'active'),
                                ('temperature_critical', 'inactive'),
                                ('temperature_warning', 'active')
                            ])
                    
                    updates['snmp_updates'].append({
                        'oid': oid,
                        'value': str(low_temp),
                        'category': category,
                        'field_name': field_name
                    })
                except ValueError as e:
                    print(f"Error processing low temperature trigger: {str(e)}")
                    updates['error'] = True
                    
            elif field_name == "Triggered Status [Temp.]":
                try:
                    triggered = int(float(value)) == 1
                    updates['state_updates'].extend([
                        ('temperature_state', 'critical_high' if triggered else 'normal'),
                        ('temperature_critical_high', 'active' if triggered else 'inactive'),
                        ('temperature_warning_high', 'inactive'),
                        ('temperature_critical_low', 'inactive'),
                        ('temperature_warning_low', 'inactive'),
                        ('temperature_critical', 'active' if triggered else 'inactive'),
                        ('temperature_warning', 'inactive')
                    ])
                    
                    updates['snmp_updates'].append({
                        'oid': oid,
                        'value': int(float(value)),
                        'category': category,
                        'field_name': field_name
                    })
                except Exception as e:
                    print(f"Error processing temperature triggered status: {str(e)}")
                    updates['error'] = True
        
        elif category == "Humidity":
            if field_name == "High Humid. Trigger Set Point":
                try:
                    high_humid = float(value)
                    # Add to snmp_updates
                    updates['snmp_updates'].append({
                        'oid': oid,
                        'value': str(high_humid),
                        'category': category,
                        'field_name': field_name
                    })
                    # Update the critical high trigger point state
                    updates['state_updates'].extend([
                        ('humidity_high_critical', str(high_humid)),  # This matches devices_to_vis_elements
                        ('humidity_high_warning', str(high_humid - 5))  # For warning threshold
                    ])

                    # Get current humidity for comparison
                    try:
                        current_humid_result = await snmp_get(device.ip, HUMIDITY_OIDS['Live Humidity'], device.community)
                        current_humid = float(current_humid_result.replace('% RH', '').strip()) if not current_humid_result.startswith("Error") else None
                    except Exception as e:
                        print(f"Error getting current humidity: {str(e)}")
                        current_humid = None
                    
                    if current_humid is not None:
                        # Calculate states based on new high trigger and current humidity
                        if current_humid >= high_humid:
                            updates['state_updates'].extend([
                                ('humidity_state', 'critical_high'),
                                ('humidity_critical_high', 'active'),
                                ('humidity_warning_high', 'inactive'),
                                ('humidity_critical_low', 'inactive'),
                                ('humidity_warning_low', 'inactive'),
                                ('humidity_critical', 'active'),
                                ('humidity_warning', 'inactive')
                            ])
                        elif current_humid >= (high_humid - 5):
                            updates['state_updates'].extend([
                                ('humidity_state', 'warning_high'),
                                ('humidity_critical_high', 'inactive'),
                                ('humidity_warning_high', 'active'),
                                ('humidity_critical_low', 'inactive'),
                                ('humidity_warning_low', 'inactive'),
                                ('humidity_critical', 'inactive'),
                                ('humidity_warning', 'active')
                            ])
                    
                    updates['snmp_updates'].append({
                        'oid': oid,
                        'value': str(high_humid),
                        'category': category,
                        'field_name': field_name
                    })
                except ValueError as e:
                    print(f"Error processing high humidity trigger: {str(e)}")
                    updates['error'] = True
                    
            elif field_name == "Low Humid. Trigger Set Point":
                try:
                    low_humid = float(value)
                    # Add to snmp_updates
                    updates['snmp_updates'].append({
                        'oid': oid,
                        'value': str(low_humid),
                        'category': category,
                        'field_name': field_name
                    })
                    # Update the critical low trigger point state
                    updates['state_updates'].extend([
                        ('humidity_low_critical', str(low_humid)),  # This matches devices_to_vis_elements
                        ('humidity_low_warning', str(low_humid + 5))  # For warning threshold
                    ])
                    # Get current humidity for comparison
                    try:
                        current_humid_result = await snmp_get(device.ip, HUMIDITY_OIDS['Live Humidity'], device.community)
                        current_humid = float(current_humid_result.replace('% RH', '').strip()) if not current_humid_result.startswith("Error") else None
                    except Exception as e:
                        print(f"Error getting current humidity: {str(e)}")
                        current_humid = None
                    
                    if current_humid is not None:
                        # Calculate states based on new low trigger and current humidity
                        if current_humid <= low_humid:
                            updates['state_updates'].extend([
                                ('humidity_state', 'critical_low'),
                                ('humidity_critical_high', 'inactive'),
                                ('humidity_warning_high', 'inactive'),
                                ('humidity_critical_low', 'active'),
                                ('humidity_warning_low', 'inactive'),
                                ('humidity_critical', 'active'),
                                ('humidity_warning', 'inactive')
                            ])
                        elif current_humid <= (low_humid + 5):
                            updates['state_updates'].extend([
                                ('humidity_state', 'warning_low'),
                                ('humidity_critical_high', 'inactive'),
                                ('humidity_warning_high', 'inactive'),
                                ('humidity_critical_low', 'inactive'),
                                ('humidity_warning_low', 'active'),
                                ('humidity_critical', 'inactive'),
                                ('humidity_warning', 'active')
                            ])
                    
                    updates['snmp_updates'].append({
                        'oid': oid,
                        'value': str(low_humid),
                        'category': category,
                        'field_name': field_name
                    })
                except ValueError as e:
                    print(f"Error processing low humidity trigger: {str(e)}")
                    updates['error'] = True
                    
            elif field_name == "Triggered Status [Humid.]":
                try:
                    triggered = int(float(value)) == 1
                    updates['state_updates'].extend([
                        ('humidity_state', 'critical_high' if triggered else 'normal'),
                        ('humidity_critical_high', 'active' if triggered else 'inactive'),
                        ('humidity_warning_high', 'inactive'),
                        ('humidity_critical_low', 'inactive'),
                        ('humidity_warning_low', 'inactive'),
                        ('humidity_critical', 'active' if triggered else 'inactive'),
                        ('humidity_warning', 'inactive')
                    ])
                    
                    updates['snmp_updates'].append({
                        'oid': oid,
                        'value': int(float(value)),
                        'category': category,
                        'field_name': field_name
                    })
                except Exception as e:
                    print(f"Error processing humidity triggered status: {str(e)}")
                    updates['error'] = True
                    
        
        #print(f"Final updates before return: {updates}")  # Debug
        return updates
            
    except Exception as e:
        print(f"Error calculating device states: {str(e)}")
        return {
            'value': str(e),
            'error': True,
            'snmp_updates': [],
            'state_updates': []
        }
        
def apply_device_updates(device: Device, updates: Dict):
    """Apply all device updates in main thread"""
    print(f"\nApplying updates for {device.name}")  # Debug
    try:
        
        # Get current states before update
        old_states = db.get_all_device_states(device.node_id)
        #print(f"States before update: {old_states}")  # Debug
        
        # Apply state updates that affect SVG
        for state_type, state_value in updates.get('state_updates', []):
            #print(f"Applying state update: {state_type} = {state_value}") # Debug
            db.update_device_state(device.node_id, state_type, state_value)
        
        # Force device states to update in memory
        st.session_state.devices = [d for d in st.session_state.devices]  # Trigger rerender
        #print("Device states refreshed in memory")  # Debug
        
        # Get updated states
        new_states = db.get_all_device_states(device.node_id)
        #print(f"States after update: {new_states}")  # Debug
        
        # Generate new vis elements with timestamp key
        current_time = str(time.time())
        nodes, edges = devices_to_vis_elements(st.session_state.devices)
        
        # Update network elements with key
        if 'network_elements' not in st.session_state:
            st.session_state.network_elements = {}
        st.session_state.network_elements = {
            'nodes': nodes,
            'edges': edges,
            'key': current_time
        }
        print(f"Updated network elements with key: {current_time}")  # Debug
            
    except Exception as e:
        print(f"Error applying device updates: {str(e)}")                          

def apply_batch_updates(updates_list, poller):
    """Apply multiple device updates in a single batch to reduce database writes and UI updates"""
    print(f"\nApplying batch updates for {len(updates_list)} devices")
    try:
        # Track all device IDs that need updates
        device_ids = set(update['device_id'] for update in updates_list)
        
        # Get all old states at once
        old_states = {device_id: db.get_all_device_states(device_id) for device_id in device_ids}
        
        # Group all state updates by device
        state_updates = {}
        timestamp_updates = {}
        for update in updates_list:
            device_id = update['device_id']
            if device_id not in state_updates:
                state_updates[device_id] = []
            state_updates[device_id].extend(update.get('state_updates', []))
            timestamp_updates[device_id] = update['timestamp']
        
        # Apply all state updates to database in one pass per device
        for device_id, updates in state_updates.items():
            for state_type, state_value in updates:
                db.update_device_state(device_id, state_type, state_value)
        
        # Update timestamps
        poller.last_poll_time.update(timestamp_updates)
        
        # Force device states to update in memory
        st.session_state.devices = [d for d in st.session_state.devices]
        
        # Get all new states at once
        new_states = {device_id: db.get_all_device_states(device_id) for device_id in device_ids}
        
        # Generate new vis elements once for all updates
        current_time = str(time.time())
        nodes, edges = devices_to_vis_elements(st.session_state.devices)
        
        # Update network elements with key
        if 'network_elements' not in st.session_state:
            st.session_state.network_elements = {}
        st.session_state.network_elements = {
            'nodes': nodes,
            'edges': edges,
            'key': current_time
        }
        print(f"Updated network elements with key: {current_time}")
        
        # Return true if any states changed
        return any(old_states[device_id] != new_states[device_id] for device_id in device_ids)
        
    except Exception as e:
        print(f"Error in batch updates: {str(e)}")
        return False

def add_device_to_lists(device: Device) -> bool:
    """Add device to topology and save state"""
    if not any(d.ip == device.ip for d in st.session_state.devices):
        # Save to database with explicit position
        db.save_device({
            'node_id': device.node_id,
            'name': device.name,
            'ip': device.ip,
            'is_snmp': device.is_snmp,
            'community': device.community,
            'write_community': device.write_community,
            'version': device.version,
            'position': device.position
        })
        
        # Add to session state
        st.session_state.devices.append(device)
        return True
    return False


def load_devices():
    """Load devices from database"""
    devices_data = db.get_all_devices()
    st.session_state.devices = [Device.from_dict(data) for data in devices_data]


def device_discovery_page():
    """Create and display the device discovery page with enhanced state tracking"""
    st.title("Device Discovery")
    
    col1, col2 = st.columns(2)
    with col1:
        start_ip = st.text_input("Start IP Address", "192.168.1.1", key="start_ip_input")
    with col2:
        end_ip = st.text_input("End IP Address", "192.168.1.254", key="end_ip_input")
    
    # Manual device addition with enhanced settings
    st.subheader("Add Device Manually")
    with st.form("manual_device_discovery"):
        name = st.text_input("Device Name", key="name_input_discovery")
        ip = st.text_input("IP Address", key="ip_input_discovery")
        is_snmp = st.checkbox("SNMP Device", value=True, key="snmp_check_discovery")
        
        if is_snmp:
            with st.expander("SNMP Settings", expanded=True):
                community = st.text_input("Read Community", value="public", key="community_input_discovery")
                write_community = st.text_input("Write Community", value="private", key="write_community_input_discovery")
                version = st.selectbox("SNMP Version", ["v1", "v2c"], key="version_select_discovery")
                simulation_mode = st.checkbox("Enable Simulation Mode", value=False, help="Generate simulated sensor readings for testing")
        else:
            community = "public"
            write_community = "private"
            version = "v2c"
            simulation_mode = False
            
        if st.form_submit_button("Add Device"):
            if name and ip:
                try:
                    ipaddress.ip_address(ip)
                    if any(d.ip == ip for d in st.session_state.devices):
                        st.error("Device with this IP already exists!")
                    else:
                        new_device = Device(
                            ip=ip,
                            name=name,
                            is_snmp=is_snmp,
                            community=community,
                            write_community=write_community,
                            version=version,
                            position={'x': 100 + len(st.session_state.devices) * 100, 
                                    'y': 100 + len(st.session_state.devices) * 50},
                            simulation_mode = simulation_mode
                        )
                        
                        # Save to database with enhanced data
                        db.save_device({
                            'node_id': new_device.node_id,
                            'name': new_device.name,
                            'ip': new_device.ip,
                            'is_snmp': new_device.is_snmp,
                            'community': new_device.community,
                            'write_community': new_device.write_community,
                            'version': new_device.version,
                            'position': new_device.position, 
                            'simulation_mode': new_device.simulation_mode
                        })
                        
                        # Initialize device states in database
                        db.update_device_state(new_device.node_id, 'temperature_critical', 'inactive')
                        db.update_device_state(new_device.node_id, 'temperature_warning', 'inactive')
                        db.update_device_state(new_device.node_id, 'door_open', 'inactive')
                        
                        st.session_state.devices.append(new_device)
                        cache_key = f"{ip}_{community}"
                        st.session_state.device_status_cache[cache_key] = True
                        
                        st.success(f"Device {name} ({ip}) added successfully!")
                        st.rerun()
                except ValueError:
                    st.error("Invalid IP address format")
            else:
                st.error("Please fill in both name and IP address")
    
    # Network scanning with enhanced device processing
    st.subheader("Network Scan")
    if st.button("Start Network Scan", key="start_scan_button"):
        try:
            ipaddress.ip_address(start_ip)
            ipaddress.ip_address(end_ip)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Scanning network...")
            new_devices = asyncio.run(scan_network(start_ip, end_ip, progress_bar))
            
            if 'devices' not in st.session_state:
                st.session_state.devices = []
            
            for device in new_devices:
                if not any(d.ip == device.ip for d in st.session_state.devices):
                    # Initialize write community for discovered devices
                    device.write_community = "private"
                    
                    # Save to database with enhanced data
                    db.save_device({
                        'node_id': device.node_id,
                        'name': device.name,
                        'ip': device.ip,
                        'is_snmp': device.is_snmp,
                        'community': device.community,
                        'write_community': device.write_community,
                        'version': device.version,
                        'position': device.position
                    })
                    
                    # Initialize device states
                    db.update_device_state(device.node_id, 'temperature_critical', 'inactive')
                    db.update_device_state(device.node_id, 'temperature_warning', 'inactive')
                    db.update_device_state(device.node_id, 'door_open', 'inactive')
                    
                    st.session_state.devices.append(device)
                    cache_key = f"{device.ip}_{device.community}"
                    st.session_state.device_status_cache[cache_key] = True
            
            progress_bar.progress(1.0)
            status_text.text("Scan complete!")
            st.success(f"Found {len(new_devices)} new SNMP devices!")
            
            if new_devices:
                st.subheader("Discovered Devices")
                device_data = [
                    {
                        "Name": device.name,
                        "IP Address": device.ip,
                        "Type": "SNMP" if device.is_snmp else "Non-SNMP",
                        "Status": "Active" if st.session_state.device_status_cache.get(f"{device.ip}_{device.community}", False) else "Inactive"
                    }
                    for device in new_devices
                ]
                st.table(device_data)
                
                # Show initial states
                st.subheader("Device States")
                for device in new_devices:
                    with st.expander(f"{device.name} ({device.ip})", expanded=False):
                        states = db.get_all_device_states(device.node_id)
                        for state_type, state_value in states.items():
                            st.text(f"{state_type.replace('_', ' ').title()}: {state_value}")
                
        except ValueError:
            st.error("Invalid IP address format")
        except Exception as e:
            st.error(f"Error during scan: {str(e)}")
    
    # Current devices list with enhanced status display
    st.subheader("Current Devices")
    if hasattr(st.session_state, 'devices') and st.session_state.devices:
        # Initialize confirmation states if not exists
        if 'device_removal_confirms' not in st.session_state:
            st.session_state.device_removal_confirms = {}
        
        for i, device in enumerate(st.session_state.devices):
            # Create an expander for each device
            with st.expander(f"{device.name} ({device.ip})", expanded=False):
                # Display device details
                st.markdown(f"**Type:** {'SNMP' if device.is_snmp else 'Non-SNMP'}")
                if device.is_snmp:
                    st.markdown(f"**Community:** {device.community}")
                    st.markdown(f"**Version:** {device.version}")
                
                # Show device states
                states = db.get_all_device_states(device.node_id)
                if states:
                    st.markdown("**Current States:**")
                    for state_type, state_value in states.items():
                        if state_value == 'active':
                            if 'temperature' in state_type:
                                st.error(f"{state_type.replace('_', ' ').title()}")
                            else:
                                st.warning(f"{state_type.replace('_', ' ').title()}")
                
                # Add remove button with confirmation
                col1, col2 = st.columns([3,1])
                
                # Initialize confirmation state for this device
                confirm_key = f"confirm_remove_{device.node_id}"
                if confirm_key not in st.session_state.device_removal_confirms:
                    st.session_state.device_removal_confirms[confirm_key] = False
                
                with col1:
                    st.session_state.device_removal_confirms[confirm_key] = st.checkbox(
                        "Confirm removal",
                        key=f"checkbox_{confirm_key}",
                        value=st.session_state.device_removal_confirms[confirm_key]
                    )
                
                with col2:
                    if st.button("Remove", key=f"remove_btn_{device.node_id}"):
                        if st.session_state.device_removal_confirms[confirm_key]:
                            try:
                                # Remove from database first
                                db.remove_device(device.node_id)
                                
                                # Remove from session state
                                st.session_state.devices.remove(device)
                                
                                # Clean up associated data
                                if device.node_id in st.session_state.device_status_history:
                                    del st.session_state.device_status_history[device.node_id]
                                if device.node_id in st.session_state.last_update_times:
                                    del st.session_state.last_update_times[device.node_id]
                                
                                # Clear cache for removed device
                                cache_key = f"{device.ip}_{device.community}"
                                if cache_key in st.session_state.device_status_cache:
                                    del st.session_state.device_status_cache[cache_key]
                                
                                st.success(f"Device {device.name} removed successfully!")
                                st.session_state.device_removal_confirms[confirm_key] = False
                                time.sleep(0.5)  # Brief pause to show success message
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error removing device: {str(e)}")
                        else:
                            st.warning("Please confirm removal by checking the box")
    else:
        st.info("No devices currently added")


# Enhanced save topology function
def save_topology():
    """Save complete topology state to file"""
    if st.session_state.devices:
        # Create a complete state snapshot
        save_data = {
            'devices': [device.to_dict() for device in st.session_state.devices],
            'device_positions': st.session_state.device_positions,
            'background_image': st.session_state.get('background_image'),
            'background_dimensions': st.session_state.get('background_dimensions')
        }
        
        # Offer download
        st.download_button(
            "Save Topology",
            data=json.dumps(save_data, indent=2),
            file_name="topology.json",
            mime="application/json"
        )

def load_topology(uploaded_file):
    """Load topology state from file"""
    try:
        topology_data = json.loads(uploaded_file.getvalue())
        
        # Clear existing state
        st.session_state.devices = []
        st.session_state.device_positions = {}
        
        # Load devices first
        devices = []
        for device_data in topology_data.get('devices', []):
            device = Device.from_dict(device_data)
            devices.append(device)
        
        # Update session state
        st.session_state.devices = devices
        st.session_state.device_positions = topology_data.get('device_positions', {})
        
        # Handle background if present
        if 'background_image' in topology_data:
            st.session_state.background_image = topology_data['background_image']
            st.session_state.background_dimensions = topology_data.get(
                'background_dimensions', 
                {'width': 1280, 'height': 1000}
            )
        
        st.success("Topology loaded successfully")
        return True
    except Exception as e:
        st.error(f"Error loading topology: {str(e)}")
        return False


# In your background image upload handler:
def handle_background_upload(uploaded_file):
    """Handle background image upload and processing"""
    try:
        image = Image.open(uploaded_file)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Store image dimensions
        st.session_state.background_dimensions = {
            'width': image.width,
            'height': image.height
        }
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        st.session_state.background_image = img_str
        
        # Save state after background update
        topology_data = {
            'devices': [device.to_dict() for device in st.session_state.devices],
            'background_image': st.session_state.background_image,
            'background_dimensions': st.session_state.background_dimensions
        }
        
        return True
        
    except Exception as e:
        st.error(f"Error uploading background image: {str(e)}")
        return False


def initialize_session_state():
    """Initialize all session state variables"""
    if 'devices' not in st.session_state:
        st.session_state.devices = []
    if 'background_image' not in st.session_state:
        st.session_state.background_image = None
    if 'background_dimensions' not in st.session_state:
        st.session_state.background_dimensions = {'width': 1280, 'height': 1000}
    if 'loaded_from_file' not in st.session_state:
        st.session_state.loaded_from_file = False
        
    # Add these new lines
    if 'expander_states' not in st.session_state:
        st.session_state.expander_states = {}
    if 'query_active' not in st.session_state:
        st.session_state.query_active = {}
    if 'selected_device_id' not in st.session_state:
        st.session_state.selected_device_id = None
    if 'last_viewport_save' not in st.session_state:
        st.session_state.last_viewport_save = 0
    if 'viewport_debounce_ms' not in st.session_state:
        st.session_state.viewport_debounce_ms = 500  # Debounce time in milliseconds


def save_current_positions():
    """Save all current positions to database"""
    if hasattr(st.session_state, 'devices'):
        for device in st.session_state.devices:
            if hasattr(device, 'position'):
                print(f"Saving position for {device.name} before switch: {device.position}")  # Debug
                db.save_device_position(device.node_id, device.position['x'], device.position['y'])

# Initialize session state and load saved positions at startup
if 'positions_cache' not in st.session_state:
    st.session_state.positions_cache = {}

async def batch_update_positions():
    """Update positions in batches to reduce database writes"""
    if not st.session_state.position_update_queue:
        return
        
    current_time = time.time()
    # Only update if 500ms has passed since last update and we have changes
    if current_time - st.session_state.last_position_update >= 0.5:
        try:
            updates = st.session_state.position_update_queue.copy()
            st.session_state.position_update_queue.clear()
            
            # Batch update all positions at once
            for device_id, position in updates.items():
                device = next((d for d in st.session_state['devices'] if d.id == device_id), None)
                if device:
                    device.position = position
                    await db.update_device_position(device_id, position)
            
            st.session_state.last_position_update = current_time
        except Exception as e:
            print(f"Error updating positions: {e}")
            # Restore failed updates to queue
            st.session_state.position_update_queue.update(updates)

def queue_position_update(device_id: str, position: Dict[str, float]):
    """Queue a position update for batch processing"""
    st.session_state.position_update_queue[device_id] = position

def main():
    """Main application function with complete feature integration"""
    
    # Initialize States
    initialize_position_state()
    
    if 'page_state' not in st.session_state:
        st.session_state.page_state = {}
    if 'device_filters' not in st.session_state:
        st.session_state.device_filters = {
            'show_snmp': True,
            'show_non_snmp': True,
            'status_filter': 'all'
        }
    if 'show_query_details' not in st.session_state:
        st.session_state.show_query_details = False
    if 'show_monitor_details' not in st.session_state:
        st.session_state.show_monitor_details = False

    if 'snmp_menu_states' not in st.session_state:
        st.session_state.snmp_menu_states = {}
        
    if 'app_initialized' not in st.session_state:
        initialize_app_state()
        # Initialize poller if needed
        if 'snmp_poller' not in st.session_state:
            st.session_state.snmp_poller = SNMPPoller()
        st.session_state.app_initialized = True
    
    # Navigation with state preservation
    page = st.sidebar.radio(
        "Navigation",
        ["Network Topology", "Device Discovery"],
        key="navigation"
    )
    
    # Save positions before page switch
    save_current_positions()
    
    # Main Content
    if page == "Network Topology":
        create_topology_view()
        
        # Device Management Sidebar
        with st.sidebar:
            st.markdown("---")
            st.subheader("Device Management")
            
            # Device Filtering
            with st.expander("Filter Devices", expanded=False):
                st.session_state.device_filters['show_snmp'] = st.checkbox(
                    "Show SNMP Devices",
                    value=st.session_state.device_filters['show_snmp']
                )
                st.session_state.device_filters['show_non_snmp'] = st.checkbox(
                    "Show Non-SNMP Devices",
                    value=st.session_state.device_filters['show_non_snmp']
                )
                st.session_state.device_filters['status_filter'] = st.selectbox(
                    "Status Filter",
                    ['all', 'active', 'inactive']
                )

    else:
        device_discovery_page()
    
    # Statistics
    with st.sidebar:
        
        # Statistics Display
        st.markdown("---")
        st.subheader("Network Statistics")
        
        if hasattr(st.session_state, 'devices'):
            total_devices = len(st.session_state.devices)
            snmp_devices = sum(1 for d in st.session_state.devices if d.is_snmp)
            active_devices = 0# will eventually implement get_active_devices_count() which works off of snmp polling
            
            stats_container = st.container()
            with stats_container:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Devices", total_devices)
                    st.metric("SNMP Devices", snmp_devices)
                with col2:
                    st.metric("Active Devices", active_devices)
                    st.metric("Other Devices", total_devices - snmp_devices)
        

if __name__ == "__main__":
    main()