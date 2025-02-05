from funcrunner.app import FunctionRunner


# Example functions to register
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle"""
    return length * width


def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email to the specified address"""
    # Implementation would go here
    return True


async def main():
    # Initialize the runner
    runner = FunctionRunner()

    # Register available functions
    runner.register_function(calculate_area, "Calculate the area of a rectangle given length and width")
    runner.register_function(send_email, "Send an email to a specified address with subject and body")

    # Process a natural language request
    result = await runner.process_llm_request("Calculate the area of a rectangle that is 5 meters by 3 meters")
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
