#!/usr/bin/env python3
"""
Quickstart Guide for ONNX Text Generation

This script demonstrates the basic usage of the OnnxTextGenerator class
for text generation using ONNX Runtime on CPU.
"""

from inference import OnnxTextGenerator
import sys


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def example_basic_generation() -> None:
    """Example 1: Basic text generation with default model."""
    print_section("Example 1: Basic Text Generation")

    # Initialize the generator with default model (Gemma-3-270m-it-ONNX)
    print("Initializing OnnxTextGenerator with default model...")
    generator = OnnxTextGenerator()
    print("✓ Model loaded successfully!\n")

    # Generate text from a prompt
    prompt = "The future of artificial intelligence is"
    print(f"Prompt: '{prompt}'")
    print("Generating...\n")

    result = generator.generate(prompt, max_new_tokens=30, temperature=0.7)
    print(f"Generated text:\n{prompt}{result['generated_text']}\n")


def example_streaming_generation() -> None:
    """Example 2: Streaming text generation (token-by-token)."""
    print_section("Example 2: Streaming Generation")

    generator = OnnxTextGenerator()

    prompt = "Once upon a time, in a land far away,"
    print(f"Prompt: '{prompt}'")
    print("Streaming output: ", end="", flush=True)

    # Stream tokens one by one
    for chunk, _ in generator.stream_generate(
        prompt, max_new_tokens=25, temperature=0.8
    ):
        print(chunk, end="", flush=True)

    print("\n")


def example_different_temperatures() -> None:
    """Example 3: Comparing different temperature settings."""
    print_section("Example 3: Temperature Comparison")

    generator = OnnxTextGenerator()
    prompt = "The best programming language is"

    temperatures = [0.0, 0.5, 1.0]

    for temp in temperatures:
        result = generator.generate(prompt, max_new_tokens=15, temperature=temp)
        temp_name = "Greedy (deterministic)" if temp == 0 else "Sampling (creative)"
        print(f"Temperature={temp} ({temp_name}):")
        print(f"  → {result['generated_text']}\n")


def example_code_completion() -> None:
    """Example 4: Code completion."""
    print_section("Example 4: Code Completion")

    # Using SmolLM2 which is better at code
    print("Initializing with SmolLM2-135M (better for code)...")
    generator = OnnxTextGenerator(
        model_id="onnx-community/SmolLM2-135M-ONNX",
        onnx_file="onnx/model.onnx",
        allow_patterns=["onnx/model.onnx*", "*.json"],
    )
    print("✓ Model loaded!\n")

    prompt = 'def calculate_fibonacci(n):\n    """\n    Calculate the nth Fibonacci number.\n    """\n    '
    print(f"Code prompt:\n{prompt}")
    print("Completion:")

    result = generator.generate(prompt, max_new_tokens=40, temperature=0.3)
    print(f"{result['generated_text']}\n")


def example_custom_parameters() -> None:
    """Example 5: Using custom generation parameters."""
    print_section("Example 5: Custom Parameters")

    generator = OnnxTextGenerator()

    prompt = "Write a haiku about coding:"
    print(f"Prompt: '{prompt}'\n")

    # Higher temperature for more creativity, lower top_p for more focus
    result = generator.generate(
        prompt,
        max_new_tokens=25,
        temperature=0.9,  # More creative
        top_p=0.85,  # Nucleus sampling
    )

    print(f"Generated (temp=0.9, top_p=0.85):\n{result['generated_text']}\n")


def main() -> None:
    """Run all examples or specific ones based on command-line arguments."""
    print("\n" + "=" * 60)
    print("  ONNX Text Generation - Quickstart Examples")
    print("=" * 60)

    # Check if user wants specific examples
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        examples = {
            "1": example_basic_generation,
            "basic": example_basic_generation,
            "2": example_streaming_generation,
            "stream": example_streaming_generation,
            "3": example_different_temperatures,
            "temperature": example_different_temperatures,
            "temp": example_different_temperatures,
            "4": example_code_completion,
            "code": example_code_completion,
            "5": example_custom_parameters,
            "custom": example_custom_parameters,
        }

        if arg in examples:
            examples[arg]()
        else:
            print(f"\nUnknown example: {arg}")
            print("\nAvailable examples:")
            print("  1 or basic  - Basic text generation")
            print("  2 or stream - Streaming generation")
            print("  3 or temp   - Temperature comparison")
            print("  4 or code   - Code completion")
            print("  5 or custom - Custom parameters")
            print("\nUsage: python quickstart.py [example_name]")
    else:
        # Run all examples sequentially
        try:
            example_basic_generation()
            example_streaming_generation()
            example_different_temperatures()
            example_code_completion()
            example_custom_parameters()

        except KeyboardInterrupt:
            print("\n\nExecution interrupted. Goodbye! 👋")
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  For more examples, see test_inference.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
