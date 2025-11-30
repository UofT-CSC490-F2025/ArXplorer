#!/usr/bin/env python3
"""
Startup script for ArXplorer frontend.

Usage:
    python run_frontend.py                    # Development mode
    python run_frontend.py --production       # Production mode
    python run_frontend.py --port 8080        # Custom port
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from frontend.app import app, initialize_search_system
from frontend.config import config


def main():
    parser = argparse.ArgumentParser(description='Run ArXplorer Frontend')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')
    parser.add_argument('--config-path', help='Path to backend config.yaml file')
    
    args = parser.parse_args()
    
    # Set configuration
    if args.production:
        app.config.from_object(config['production'])
    else:
        app.config.from_object(config['development'])
    
    # Override config with command line args
    if args.no_debug:
        app.config['DEBUG'] = False
    
    if args.config_path:
        app.config['BACKEND_CONFIG_PATH'] = args.config_path
    
    print("="*60)
    print("ArXplorer Frontend")
    print("="*60)
    print(f"Mode: {'Production' if args.production else 'Development'}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {app.config['DEBUG']}")
    print(f"Backend Config: {app.config.get('BACKEND_CONFIG_PATH', 'config.yaml')}")
    print("="*60)
    
    # Initialize search system
    print("\nInitializing search system...")
    initialize_search_system()
    
    # Run the application
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")
    
    if args.production:
        # Use Waitress for production
        try:
            from waitress import serve
            serve(app, host=args.host, port=args.port, threads=4)
        except ImportError:
            print("Warning: Waitress not installed. Install with: pip install waitress")
            print("Falling back to Flask development server...")
            app.run(host=args.host, port=args.port, debug=False, threaded=True)
    else:
        # Use Flask development server
        app.run(
            host=args.host,
            port=args.port,
            debug=app.config['DEBUG'],
            threaded=True
        )


if __name__ == '__main__':
    main()