import React from 'react';
import { Network } from 'vis-network';
import { DataSet } from 'vis-data';
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import { createDeviceSVG } from './deviceRenderer';
const DEBUG = true;

interface NetworkProps {
  nodes: any[];
  edges: any[];
  options: any;
  background_image?: string;
  height?: string;
  viewport?: {
    position: {
      x: number;
      y: number;
    };
    scale: number;
  };
}

interface State {
  network: Network | null;
  nodes: DataSet<any>;
  edges: DataSet<any>;
  viewportInitialized: boolean;
  isStabilized: boolean;
}

class VisNetwork extends StreamlitComponentBase<State> {
  private container = React.createRef<HTMLDivElement>();
  private networkContainer = React.createRef<HTMLDivElement>();
  private backgroundCanvas = React.createRef<HTMLCanvasElement>();
  private backgroundImage: HTMLImageElement | null = null;
  private animationFrame: number | null = null;
  private viewportUpdateTimeout: NodeJS.Timeout | null = null;
  
  public state: State = {
    network: null,
    nodes: new DataSet([]),
    edges: new DataSet([]),
    viewportInitialized: false,
    isStabilized: false
  };

  public componentDidMount() {
    console.log("Component mounted");
    Streamlit.setFrameHeight(600);
    this.setupNetwork();
  }

  public componentDidUpdate() {
    const props = this.props.args as NetworkProps;
    
    console.log("Component update received - nodes:", props.nodes);  // Debug

    // Handle network data updates with SVG processing
    if (this.state.network) {
      const processedNodes = props.nodes.map(node => ({
        ...node,
        shape: 'image',
        image: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
          createDeviceSVG(node, node.state)
        )}`,
        size: node.size || 60  // Use node size if provided, fallback to 60
      }));

      const nodes = new DataSet(processedNodes);
      const edges = new DataSet(props.edges);
      this.state.network.setData({ nodes, edges });
    }

    // Handle background image updates
    if (props.background_image && (!this.backgroundImage || this.backgroundImage.src !== props.background_image)) {
      this.setupBackgroundImage(props.background_image);
    }
  }

  private setupNetwork() {
    if (!this.networkContainer.current) return;

    const props = this.props.args as NetworkProps;
    
    if (DEBUG) console.log('Setting up network with props:', props);
    
    // Process nodes to add SVG representation
    const processedNodes = props.nodes.map(node => ({
      ...node,
      shape: 'image',
      image: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
        createDeviceSVG(node, node.state)
      )}`,
      size: node.size || 60
    }));

    const nodes = new DataSet(processedNodes);
    const edges = new DataSet(props.edges);

    const network = new Network(
      this.networkContainer.current,
      { nodes, edges },
      props.options
    );

    // Override the fit method to prevent automatic fitting
    const originalFit = network.fit.bind(network);
    network.fit = function(...args) {
      console.log("Preventing automatic fit");
      return;
    };

    // Set initial viewport immediately
    if (props.viewport) {
      console.log("Setting initial viewport:", props.viewport);
      requestAnimationFrame(() => {
        network.moveTo({
          position: props.viewport!.position,
          scale: props.viewport!.scale,
          animation: false
        });
      });
    }

    // Selection handler - ONLY send selection
    network.on('select', (params) => {
      if (DEBUG) console.log('Select event:', params);
      const value = {
        selected: params.nodes,
        type: 'selection'
      };
      if (DEBUG) console.log('Sending select value:', value);
      Streamlit.setComponentValue(value);
    });

    // Position handler - ONLY send positions
    network.on('dragEnd', (params) => {
      if (DEBUG) console.log('DragEnd event:', params);
      if (DEBUG) console.log('Network stabilized state:', this.state.isStabilized);
      
      if (this.viewportUpdateTimeout) {
        clearTimeout(this.viewportUpdateTimeout);
      }

      // Don't wait for stabilization on dragEnd
      const positions = this.getNodePositions();
      if (DEBUG) console.log('Node positions:', positions);
      const value = {
        positions: positions,
        type: 'position'
      };
      if (DEBUG) console.log('Sending dragEnd value:', value);
      Streamlit.setComponentValue(value);
    });

    // Viewport handler - ONLY send viewport
    network.on('zoom', () => {
      if (DEBUG) console.log('Zoom event');
      if (this.viewportUpdateTimeout) {
        clearTimeout(this.viewportUpdateTimeout);
      }

      this.viewportUpdateTimeout = setTimeout(() => {
        const value = {
          viewport: {
            scale: network.getScale(),
            position: network.getViewPosition()
          },
          type: 'viewport'
        };
        if (DEBUG) console.log('Sending zoom value:', value);
        Streamlit.setComponentValue(value);
      }, 100);
    });

    // Add click handler to ensure drag events work properly
    network.on('click', () => {
      if (DEBUG) console.log('Click event');
    });

    // Add dragStart handler to ensure drag events work properly
    network.on('dragStart', () => {
      if (DEBUG) console.log('DragStart event');
    });

    // Add context menu handler
    network.on('oncontext', (params) => {
      params.event.preventDefault();
      const nodeId = this.state.network?.getNodeAt(params.pointer.DOM);
      
      if (nodeId) {
        const value = {
          nodeId: nodeId,
          position: {
            x: params.pointer.DOM.x,
            y: params.pointer.DOM.y
          },
          type: 'context_menu'
        };
        if (DEBUG) console.log('Sending context menu value:', value);
        Streamlit.setComponentValue(value);
      }
    });

    network.on('stabilized', () => {
      if (DEBUG) console.log("Network stabilized");
      this.setState({ isStabilized: true });
    });

    this.setState({ network, nodes, edges, isStabilized: true }); // Set stabilized to true immediately
  }

  private getNodePositions() {
    if (!this.state.network || !this.state.nodes) return {};
    
    const positions: { [key: string]: { x: number, y: number } } = {};
    this.state.nodes.forEach((node) => {
      const pos = this.state.network!.getPosition(node.id);
      if (DEBUG) console.log(`Position for ${node.id}:`, pos);
      positions[node.id] = { x: pos.x, y: pos.y };
    });
    
    return positions;
  }

  public componentWillUnmount() {
    if (this.state.network) {
        this.state.network.destroy();
    }
    if (this.animationFrame) {
        cancelAnimationFrame(this.animationFrame);
    }
    if (this.viewportUpdateTimeout) {
        clearTimeout(this.viewportUpdateTimeout);
    }
  }

  private setupBackgroundImage(imageUrl: string) {
    const img = new Image();
    img.onload = () => {
      console.log("Background image loaded", { width: img.width, height: img.height });
      this.backgroundImage = img;
      this.startRenderLoop();
    };
    img.onerror = (e) => {
      console.error("Error loading background image:", e);
    };
    img.src = imageUrl;
  }

  private startRenderLoop() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    this.renderBackground();
  }

  private renderBackground() {
    if (!this.backgroundCanvas.current || !this.backgroundImage || !this.state.network) return;
    
    const canvas = this.backgroundCanvas.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Get current viewport state
    const position = this.state.network.getViewPosition();
    const scale = this.state.network.getScale();

    // Ensure canvas dimensions match container
    const containerRect = canvas.getBoundingClientRect();
    canvas.width = containerRect.width;
    canvas.height = containerRect.height;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate dimensions based on scale
    const srcWidth = canvas.width / scale;
    const srcHeight = canvas.height / scale;
    const imgCenterX = ((this.backgroundImage.width / 2) - srcWidth / 2) + position.x;
    const imgCenterY = ((this.backgroundImage.height / 2) - srcHeight / 2) + position.y;

    try {
      ctx.drawImage(
        this.backgroundImage,
        imgCenterX,
        imgCenterY,
        srcWidth,
        srcHeight,
        0,
        0,
        canvas.width,
        canvas.height
      );
    } catch (error) {
      console.error("Error rendering background:", error);
    }

    this.animationFrame = requestAnimationFrame(() => this.renderBackground());
  }

  public render() {
    return (
      <div
        ref={this.container}
        style={{
          width: '100%',
          height: '600px',
          position: 'relative',
          border: '1px solid #ddd',
          overflow: 'hidden'
        }}
      >
        <canvas
          ref={this.backgroundCanvas}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: 0,
            opacity: 0.3
          }}
        />
        <div 
          ref={this.networkContainer}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: 1
          }}
        />
      </div>
    );
  }
}

export default withStreamlitConnection(VisNetwork);