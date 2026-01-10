#!/usr/bin/env python3
"""
Quickstart Guide for ONNX Text Generation

This script demonstrates the basic usage of the OnnxTextGenerator class
for text generation using ONNX Runtime on CPU.
"""

import argparse

from inference import OnnxTextGenerator


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def example_basic_generation(generator: OnnxTextGenerator) -> None:
    """Example 1: Basic text generation."""
    print_section("Example 1: Basic Text Generation")

    # Generate text from a prompt
    prompt = "The future of artificial intelligence is"
    print(f"Prompt: '{prompt}'")
    print("Generating...\n")

    result = generator.generate(prompt, max_new_tokens=30, temperature=0.7)
    print(f"Generated text:\n{prompt}{result['generated_text']}\n")


def example_streaming_generation(generator: OnnxTextGenerator) -> None:
    """Example 2: Streaming text generation (token-by-token)."""
    print_section("Example 2: Streaming Generation")

    prompt = "Once upon a time, in a land far away,"
    print(f"Prompt: '{prompt}'")
    print("Streaming output: ", end="", flush=True)

    # Stream tokens one by one
    for chunk, _ in generator.stream_generate(
        prompt, max_new_tokens=25, temperature=0.8
    ):
        print(chunk, end="", flush=True)

    print("\n")


def example_different_temperatures(generator: OnnxTextGenerator) -> None:
    """Example 3: Comparing different temperature settings."""
    print_section("Example 3: Temperature Comparison")

    prompt = "The best programming language is"
    print(f"Prompt: '{prompt}'")

    # Run only 2 temps to save time if running sequentially
    temperatures = [0.0, 0.7]

    for temp in temperatures:
        result = generator.generate(prompt, max_new_tokens=15, temperature=temp)
        temp_name = "Greedy" if temp == 0 else "Creative"
        print(f"\nTemperature={temp} ({temp_name}):")
        print(f"  → {prompt}{result['generated_text']}")


def example_code_completion(generator: OnnxTextGenerator) -> None:
    """Example 4: Code completion."""
    print_section("Example 4: Code Completion")

    prompt = 'def calculate_fibonacci(n):\n    """\n    Calculate the nth Fibonacci number.\n    """\n    '
    print(f"Code prompt:\n{prompt}")
    print("Completion:")

    result = generator.generate(prompt, max_new_tokens=40, temperature=0.2)
    print(f"{result['generated_text']}\n")


def example_custom_parameters(generator: OnnxTextGenerator) -> None:
    """Example 5: Using custom generation parameters."""
    print_section("Example 5: Custom Parameters")

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


def example_beam_search(generator: OnnxTextGenerator) -> None:
    """Example 6: Using beam search for more deterministic, higher quality results."""
    print_section("Example 6: Beam Search")

    prompt = "The three laws of robotics are:"
    print(f"Prompt: '{prompt}'")
    print("Generating with 5 beams...\n")

    # Beam search usually results in higher quality but more deterministic output
    result = generator.generate(prompt, max_new_tokens=50, num_beams=5)
    print(f"Generated text:\n{prompt}{result['generated_text']}\n")


def main() -> None:
    """Run examples with optional model override."""
    parser = argparse.ArgumentParser(description="ONNX Text Generation Quickstart")
    parser.add_argument(
        "example",
        nargs="?",
        default="all",
        help="Specific example to run (1-6, or 'all')",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Hugging Face model ID to use (overrides default)",
    )
    parser.add_argument(
        "--onnx-file",
        "-f",
        default=None,
        help="Specific ONNX file to use (e.g. model_q4f16.onnx). If None, downloads all ONNX files.",
    )
    parser.add_argument(
        "--execution-provider",
        "-ep",
        default=None,
        help="Execution provider to use (default: auto-selection)",
    )
    parser.add_argument(
        "--num-beams",
        "-b",
        type=int,
        default=1,
        help="Number of beams for search (default: 1, which is greedy/sampling)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ONNX Text Generation - Quickstart Examples")
    print("=" * 60)

    if args.model:
        print(f"Using custom model: {args.model}")
    else:
        print("Using default model")

    # Initialize generator once
    try:
        generator = OnnxTextGenerator(
            model_id=args.model,
            onnx_file=args.onnx_file,
            execution_providers=args.execution_provider,
            num_beams=args.num_beams,
        )
        print(
            f"✓ Model loaded successfully using {generator.execution_provider} (beams={args.num_beams})!\n"
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Map examples
    examples = {
        "1": example_basic_generation,
        "basic": example_basic_generation,
        "2": example_streaming_generation,
        "stream": example_streaming_generation,
        "3": example_different_temperatures,
        "temp": example_different_temperatures,
        "4": example_code_completion,
        "code": example_code_completion,
        "5": example_custom_parameters,
        "custom": example_custom_parameters,
        "6": example_beam_search,
        "beam": example_beam_search,
    }

    if args.example.lower() == "all":
        try:
            example_basic_generation(generator)
            example_streaming_generation(generator)
            example_different_temperatures(generator)
            example_code_completion(generator)
            example_custom_parameters(generator)
            example_beam_search(generator)
        except KeyboardInterrupt:
            print("\n\nExecution interrupted. Goodbye! 👋")
    elif args.example in examples:
        examples[args.example](generator)
    else:
        print(f"\nUnknown example: {args.example}")
        print("Available examples: 1, 2, 3, 4, 5, 6, or all")


if __name__ == "__main__":
    main()
