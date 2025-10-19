"""
Visualization tool for LangGraph React Agent.
This script generates a visual representation of the agent's execution graph.
"""

from langgraph_react_agent import LangGraphReactAgent
from IPython.display import Image, display
import os


def visualize_graph(output_path: str = "langgraph_agent_flow.png"):
    """
    Generate and save a visual representation of the agent graph.
    
    Args:
        output_path: Path to save the visualization
    """
    # Initialize agent with dummy config
    llm_config = {
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 1.1
    }
    
    model_path = "../models"
    
    # Create agent
    print("Creating LangGraph React Agent...")
    agent = LangGraphReactAgent(llm_config=llm_config, model_path=model_path)
    
    # Generate graph visualization
    print(f"Generating graph visualization...")
    
    try:
        # Get the graph as a Mermaid diagram
        mermaid_png = agent.graph.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_path, "wb") as f:
            f.write(mermaid_png)
        
        print(f"✅ Graph visualization saved to: {output_path}")
        
        # Display if in Jupyter
        try:
            display(Image(mermaid_png))
        except:
            print("Not in Jupyter environment, skipping display.")
            
    except Exception as e:
        print(f"Error generating visualization: {e}")
        print("\nGenerating Mermaid markdown instead...")
        
        # Fallback to Mermaid markdown
        mermaid_md = agent.graph.get_graph().draw_mermaid()
        
        md_path = output_path.replace('.png', '.md')
        with open(md_path, 'w') as f:
            f.write("# LangGraph React Agent Flow\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_md)
            f.write("\n```\n")
        
        print(f"✅ Mermaid markdown saved to: {md_path}")
        print("\nYou can visualize this at: https://mermaid.live/")


def print_node_info():
    """Print detailed information about each node in the graph."""
    
    print("\n" + "="*80)
    print("LANGGRAPH REACT AGENT - NODE INFORMATION")
    print("="*80 + "\n")
    
    nodes = {
        "llm_call": {
            "description": "Call LLM API to generate response",
            "inputs": ["messages", "round", "num_llm_calls_available"],
            "outputs": ["messages (updated)", "round (incremented)"],
            "side_effects": ["Decrements LLM call counter", "Prints round information"]
        },
        "parse_response": {
            "description": "Parse LLM response to extract tool calls or answers",
            "inputs": ["messages"],
            "outputs": ["tool_name", "tool_args", "tool_result (on error)"],
            "side_effects": ["None"]
        },
        "execute_tool": {
            "description": "Execute the parsed tool call",
            "inputs": ["tool_name", "tool_args"],
            "outputs": ["messages (with tool_response)", "cleaned tool state"],
            "side_effects": ["Calls external tools", "Prints tool results"]
        },
        "check_termination": {
            "description": "Check all termination conditions",
            "inputs": ["messages", "round", "num_llm_calls_available", "start_time"],
            "outputs": ["should_terminate", "termination_reason", "need_final_answer"],
            "side_effects": ["Token counting", "Time checking"]
        },
        "extract_answer": {
            "description": "Extract final answer from response",
            "inputs": ["messages", "termination_reason"],
            "outputs": ["prediction", "termination"],
            "side_effects": ["Prints final results"]
        }
    }
    
    for node_name, info in nodes.items():
        print(f"📍 {node_name.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Inputs: {', '.join(info['inputs'])}")
        print(f"   Outputs: {', '.join(info['outputs'])}")
        print(f"   Side Effects: {', '.join(info['side_effects'])}")
        print()
    
    print("="*80)
    print("\nCONDITIONAL EDGES")
    print("="*80 + "\n")
    
    edges = {
        "_should_execute_tool": {
            "from": "parse_response",
            "conditions": {
                "tool_name is set": "execute_tool",
                "tool_name is None": "check_termination"
            }
        },
        "_should_continue": {
            "from": "check_termination",
            "conditions": {
                "should_terminate + need_final_answer": "llm_call",
                "should_terminate + has answer": "extract_answer",
                "should_terminate + other": "END",
                "not should_terminate": "llm_call"
            }
        }
    }
    
    for edge_name, info in edges.items():
        print(f"🔀 {edge_name}")
        print(f"   From: {info['from']}")
        print(f"   Conditions:")
        for condition, target in info['conditions'].items():
            print(f"      • {condition} → {target}")
        print()
    
    print("="*80 + "\n")


def print_execution_flow():
    """Print typical execution flows through the graph."""
    
    print("\n" + "="*80)
    print("TYPICAL EXECUTION FLOWS")
    print("="*80 + "\n")
    
    flows = {
        "Normal Tool Call Flow": [
            "llm_call",
            "parse_response (tool detected)",
            "execute_tool",
            "check_termination (continue)",
            "llm_call",
            "...",
            "parse_response (answer detected)",
            "check_termination (answer found)",
            "extract_answer",
            "END"
        ],
        "Direct Answer Flow": [
            "llm_call",
            "parse_response (no tool)",
            "check_termination (answer found)",
            "extract_answer",
            "END"
        ],
        "Token Limit Flow": [
            "llm_call",
            "parse_response",
            "check_termination (token limit reached)",
            "llm_call (generate final answer)",
            "parse_response",
            "check_termination",
            "extract_answer",
            "END"
        ],
        "Timeout Flow": [
            "llm_call",
            "...",
            "check_termination (time limit reached)",
            "END"
        ],
        "LLM Call Limit Flow": [
            "llm_call",
            "...",
            "check_termination (call limit reached)",
            "extract_answer",
            "END"
        ]
    }
    
    for flow_name, steps in flows.items():
        print(f"📋 {flow_name}")
        for i, step in enumerate(steps, 1):
            if step == "...":
                print(f"   {step} (repeated cycles)")
            else:
                print(f"   {i}. {step}")
        print()
    
    print("="*80 + "\n")


def main():
    """Main function to run all visualization utilities."""
    
    print("\n" + "="*80)
    print("LANGGRAPH REACT AGENT VISUALIZATION TOOL")
    print("="*80 + "\n")
    
    # Print node information
    print_node_info()
    
    # Print execution flows
    print_execution_flow()
    
    # Generate graph visualization
    print("\nGenerating graph visualization...")
    visualize_graph("langgraph_agent_flow.png")
    
    print("\n" + "="*80)
    print("✅ Visualization complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
