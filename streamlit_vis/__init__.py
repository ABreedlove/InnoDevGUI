import os
import streamlit.components.v1 as components
import json
from typing import Dict, List, Optional, Any

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "vis_network",
        url="http://localhost:3001",
    )
else:
    # Get the absolute path to the build directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(current_dir, "build")  # Not frontend/build, just build
    
    if not os.path.exists(build_dir):
        raise Exception(f"Build directory not found at: {build_dir}")
        
    _component_func = components.declare_component(
        "vis_network", 
        path=build_dir
    )

def vis_network(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]] = None,
    options: Dict[str, Any] = None,
    background_image: Optional[str] = None,
    updates: Optional[Dict[str, Any]] = None,  # New parameter for updates
    key: Optional[str] = None,
    height: str = "600px",
    viewport: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced vis_network component with update support
    """
    edges = edges or []
    options = options or {}
    
    default_options = {
        "physics": {
            "enabled": False
        },
        "nodes": {
            "shape": "custom",  # Enable custom shapes
            "font": {
                "size": 12,
                "multi": True
            },
            "margin": 10
        },
        "interaction": {
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "selectable": True,
            "multiselect": True
        }
    }
    
    merged_options = {**default_options, **options}
    
    # Pass updates to the component
    component_value = _component_func(
        nodes=nodes,
        edges=edges,
        options=merged_options,
        background_image=background_image,
        updates=updates,  # Include updates
        viewport=viewport,
        key=key,
        height=height
    )
    
    return component_value or {"selected": [], "positions": {}}