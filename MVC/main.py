import streamlit as st
from controller.AppController import AppController 

def main():
    """Main entry point for the application."""
    # Create the app controller
    controller = AppController()
    
    # Run the application
    controller.run()

if __name__ == "__main__":
    main()