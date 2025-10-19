"""
Script to run the LangGraph React Agent on evaluation data.
This script mirrors the functionality of run_multi_react.py but uses the LangGraph implementation.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm

from langgraph_react_agent import LangGraphReactAgent


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Run LangGraph React Agent on evaluation data')
    parser.add_argument(
        '--input',
        type=str,
        default='./eval_data/example.jsonl',
        help='Input JSONL file with questions'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output_data/langgraph_results.jsonl',
        help='Output JSONL file for results'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='../models',
        help='Path to the local model for tokenizer'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='tongyi-deepresearch',
        help='Model identifier'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Temperature for LLM generation'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p for LLM generation'
    )
    parser.add_argument(
        '--presence-penalty',
        type=float,
        default=1.1,
        help='Presence penalty for LLM generation'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Start index in the input file'
    )
    parser.add_argument(
        '--end-idx',
        type=int,
        default=None,
        help='End index in the input file (exclusive)'
    )
    
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading data from {args.input}...")
    data = load_jsonl(args.input)
    
    # Determine data range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(data)
    data = data[start_idx:end_idx]
    
    print(f"Processing {len(data)} items (from index {start_idx} to {end_idx})...")
    
    # Initialize agent
    llm_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty
    }
    
    print(f"Initializing LangGraph React Agent...")
    print(f"Model path: {args.model_path}")
    print(f"LLM config: {llm_config}")
    
    agent = LangGraphReactAgent(llm_config=llm_config, model_path=args.model_path)
    
    # Process each item
    results = []
    for idx, item in enumerate(tqdm(data, desc="Processing")):
        print(f"\n{'='*100}")
        print(f"Processing item {start_idx + idx + 1}/{start_idx + len(data)}")
        print(f"{'='*100}")
        
        try:
            # Prepare data format expected by agent
            data_item = {
                "item": item,
                "planning_port": None
            }
            
            # Run agent
            result = agent.run(data_item, model=args.model_name)
            
            # Save result
            results.append(result)
            
            # Save intermediate results after each item
            save_jsonl(results, args.output)
            print(f"\nIntermediate results saved to {args.output}")
            
        except Exception as e:
            print(f"Error processing item {start_idx + idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error result
            error_result = {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "messages": [],
                "prediction": f"Error: {str(e)}",
                "termination": "error"
            }
            results.append(error_result)
            save_jsonl(results, args.output)
    
    # Final save
    save_jsonl(results, args.output)
    print(f"\n{'='*100}")
    print(f"All results saved to {args.output}")
    print(f"Total items processed: {len(results)}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
